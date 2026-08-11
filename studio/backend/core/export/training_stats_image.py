# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Render and upload the Training History charts with every Studio Hub export."""

from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from loggers import get_logger
from storage.studio_db import get_connection

logger = get_logger(__name__)

TRAINING_STATS_FILENAME = "training_stats.png"

_BACKGROUND = "#fefefd"
_CARD = "#ffffff"
_FOREGROUND = "#202322"
_MUTED_FOREGROUND = "#7d7d7d"
_BORDER = "#e8ecea"
_GRID = "#e5e9e7"
_LOSS = "#3b82f6"
_SMOOTHED = "#f59e0b"
_GRAD_NORM = "#f97316"
_LEARNING_RATE = "#8b5cf6"
_EVAL_LOSS = "#ef4444"

_IMAGE_WIDTH = 1440
_IMAGE_HEIGHT = 760
_DPI = 160
_MAX_RENDER_POINTS = 800
_SMOOTHING = 0.6
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
_SERIES_KEYS = ("loss", "learning_rate", "grad_norm", "eval_loss")

MetricSeries = dict[str, list[tuple[float, float]]]


def _empty_series() -> MetricSeries:
    return {key: [] for key in _SERIES_KEYS}


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _resolved_path(value: Any) -> Optional[Path]:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    try:
        path = Path(value).expanduser()
        if not path.is_absolute():
            try:
                from utils.paths import resolve_output_dir

                path = resolve_output_dir(str(value))
            except (ImportError, OSError, RuntimeError, ValueError):
                pass
        return path.resolve(strict = False)
    except (OSError, RuntimeError, ValueError):
        return None


def _matching_run_id(checkpoint_path: Any) -> Optional[str]:
    """Newest Training History run whose output contains this checkpoint."""
    checkpoint = _resolved_path(checkpoint_path)
    if checkpoint is None:
        return None

    try:
        conn = get_connection()
    except Exception:
        logger.debug("Could not open Training History for export metrics", exc_info = True)
        return None
    try:
        rows = conn.execute(
            """
            SELECT id, output_dir, started_at
            FROM training_runs
            WHERE output_dir IS NOT NULL
            ORDER BY started_at DESC
            """
        ).fetchall()
    except Exception:
        logger.debug("Could not query Training History for export metrics", exc_info = True)
        return None
    finally:
        conn.close()

    matches: list[tuple[int, str, str]] = []
    for row in rows:
        output_dir = _resolved_path(row["output_dir"])
        if output_dir is None:
            continue
        if checkpoint == output_dir or output_dir in checkpoint.parents:
            matches.append((len(output_dir.parts), str(row["started_at"] or ""), str(row["id"])))
    return max(matches)[2] if matches else None


def _series_from_database(checkpoint_path: Any) -> MetricSeries:
    run_id = _matching_run_id(checkpoint_path)
    if run_id is None:
        return _empty_series()

    try:
        conn = get_connection()
    except Exception:
        logger.debug("Could not open Training History metrics for run %s", run_id, exc_info = True)
        return _empty_series()
    try:
        rows = conn.execute(
            """
            SELECT step, loss, learning_rate, grad_norm, eval_loss
            FROM training_metrics
            WHERE run_id = ?
            ORDER BY step
            """,
            (run_id,),
        ).fetchall()
    except Exception:
        logger.debug("Could not read Training History metrics for run %s", run_id, exc_info = True)
        return _empty_series()
    finally:
        conn.close()

    series = _empty_series()
    for row in rows:
        step = _finite_number(row["step"])
        if step is None or step <= 0:
            continue
        for key in _SERIES_KEYS:
            value = _finite_number(row[key])
            if value is not None:
                series[key].append((step, value))
    return series


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = _CHECKPOINT_RE.fullmatch(path.name)
    return (int(match.group(1)) if match else -1, path.name)


def _trainer_state_candidates(checkpoint_path: Any) -> Iterable[Path]:
    checkpoint = _resolved_path(checkpoint_path)
    if checkpoint is None:
        return
    if checkpoint.is_file():
        checkpoint = checkpoint.parent

    yielded: set[Path] = set()

    def emit(path: Path):
        if path not in yielded:
            yielded.add(path)
            return path
        return None

    direct = emit(checkpoint / "trainer_state.json")
    if direct is not None:
        yield direct

    if _CHECKPOINT_RE.fullmatch(checkpoint.name):
        parent = emit(checkpoint.parent / "trainer_state.json")
        if parent is not None:
            yield parent
        checkpoint = checkpoint.parent

    try:
        children = sorted(
            (child for child in checkpoint.iterdir() if child.is_dir()),
            key = _checkpoint_sort_key,
            reverse = True,
        )
    except OSError:
        children = []
    for child in children:
        if not _CHECKPOINT_RE.fullmatch(child.name):
            continue
        candidate = emit(child / "trainer_state.json")
        if candidate is not None:
            yield candidate


def _series_from_log_history(log_history: Any) -> MetricSeries:
    series = _empty_series()
    if not isinstance(log_history, list):
        return series

    # A later log for the same metric and step is what Trainer displays after a resume.
    by_key: dict[str, dict[float, float]] = {key: {} for key in _SERIES_KEYS}
    for item in log_history:
        if not isinstance(item, dict):
            continue
        step = _finite_number(item.get("step", item.get("global_step")))
        if step is None or step <= 0:
            continue
        for key in _SERIES_KEYS:
            value = _finite_number(item.get(key))
            if value is not None:
                by_key[key][step] = value

    for key, values in by_key.items():
        series[key] = sorted(values.items())
    return series


def _series_from_trainer_state(checkpoint_path: Any) -> MetricSeries:
    for candidate in _trainer_state_candidates(checkpoint_path):
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding = "utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            logger.debug("Could not read training metrics from %s", candidate, exc_info = True)
            continue
        series = _series_from_log_history(payload.get("log_history"))
        if _has_metrics(series):
            return series
    return _empty_series()


def load_training_metric_series(checkpoint_path: Any) -> MetricSeries:
    """Load exactly the four metric series consumed by Studio's history charts."""
    series = _series_from_database(checkpoint_path)
    return series if _has_metrics(series) else _series_from_trainer_state(checkpoint_path)


def _has_metrics(series: MetricSeries) -> bool:
    return any(series.get(key) for key in _SERIES_KEYS)


def _compress(data: list[Any]) -> list[Any]:
    if len(data) <= _MAX_RENDER_POINTS:
        return data
    stride = math.ceil(len(data) / _MAX_RENDER_POINTS)
    return [item for index, item in enumerate(data) if index % stride == 0 or index == len(data) - 1]


def _ema(data: list[tuple[float, float]]) -> list[tuple[float, float, float]]:
    if not data:
        return []
    is_constant = all(value == data[0][1] for _, value in data)
    last = 0.0
    count = 0
    output = []
    for step, value in data:
        if is_constant:
            smoothed = value
        else:
            last = last * _SMOOTHING + (1 - _SMOOTHING) * value
            count += 1
            smoothed = last / (1 - _SMOOTHING**count)
        output.append((step, value, smoothed))
    return output


def _domain(values: Iterable[float]) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return (0.0, 1.0)
    low, high = min(finite), max(finite)
    if low == high:
        pad = abs(low) * 0.08 if low else 0.1
    else:
        pad = (high - low) * 0.12
    return (low - pad, high + pad)


def _visible_step_domain(series: MetricSeries) -> tuple[float, float]:
    steps = sorted(
        {
            step
            for key in ("loss", "learning_rate", "grad_norm")
            for step, _value in series.get(key, [])
        }
    )
    if not steps:
        return (0.0, 1.0)
    start, end = steps[0], steps[-1]
    if start == end:
        return (start, start + 4)
    if end - start < 6:
        return (max(start, end - 6), end)
    return (start, end)


def _step_ticks(low: float, high: float, target_count: int = 6) -> list[float]:
    if not (math.isfinite(low) and math.isfinite(high)):
        return [0, 1]
    if high <= low:
        return [low, high]
    step_size = max(1, math.ceil((high - low) / (target_count - 1)))
    ticks = []
    current = low
    while current < high:
        ticks.append(current)
        current += step_size
    ticks.append(high)
    return list(dict.fromkeys(ticks))


def _strip_zeroes(value: str) -> str:
    value = value.rstrip("0").rstrip(".")
    return "0" if value in ("", "-0") else value


def _format_axis_metric(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1000:
        decimals = 0
    elif absolute >= 100:
        decimals = 1
    elif absolute >= 1:
        decimals = 3
    elif absolute >= 0.01:
        decimals = 4
    else:
        decimals = 5
    return _strip_zeroes(f"{value:.{decimals}f}")


def _format_metric(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1000:
        decimals = 0
    elif absolute >= 100:
        decimals = 2
    elif absolute >= 1:
        decimals = 4
    elif absolute >= 0.01:
        decimals = 5
    elif absolute >= 0.0001:
        decimals = 6
    else:
        decimals = 8
    return _strip_zeroes(f"{value:.{decimals}f}")


def _format_step(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(round(value))


def _format_lr_axis(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def _card_axes(fig, card: tuple[int, int, int, int], title: str):
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.transforms import IdentityTransform

    x, y, width, height = card
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle = "round,pad=0,rounding_size=25",
        transform = IdentityTransform(),
        facecolor = _CARD,
        edgecolor = _BORDER,
        linewidth = 1.25,
        zorder = -10,
        clip_on = False,
    )
    fig.add_artist(patch)
    fig.text(
        (x + 18) / _IMAGE_WIDTH,
        (y + height - 25) / _IMAGE_HEIGHT,
        title,
        color = _FOREGROUND,
        fontsize = 9,
        fontweight = 500,
        ha = "left",
        va = "center",
    )

    axes = fig.add_axes(
        [
            (x + 66) / _IMAGE_WIDTH,
            (y + 50) / _IMAGE_HEIGHT,
            (width - 86) / _IMAGE_WIDTH,
            (height - 100) / _IMAGE_HEIGHT,
        ],
        facecolor = _CARD,
    )
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.tick_params(axis = "both", which = "both", length = 0, pad = 6, labelsize = 6.5)
    axes.tick_params(colors = _MUTED_FOREGROUND)
    axes.grid(axis = "y", color = _GRID, linestyle = (0, (3, 3)), linewidth = 0.65)
    axes.set_axisbelow(True)
    return axes


def _set_common_axes(axes, x_domain: tuple[float, float], y_domain: tuple[float, float]):
    axes.set_xlim(*x_domain)
    axes.set_ylim(*y_domain)
    x_ticks = _step_ticks(*x_domain)
    axes.set_xticks(x_ticks, labels = [_format_step(value) for value in x_ticks])
    y_ticks = [y_domain[0] + (y_domain[1] - y_domain[0]) * index / 4 for index in range(5)]
    axes.set_yticks(y_ticks, labels = [_format_axis_metric(value) for value in y_ticks])


def _legend(axes, *, columns: int = 1):
    legend = axes.legend(
        loc = "upper center",
        bbox_to_anchor = (0.5, -0.18),
        ncol = columns,
        frameon = False,
        fontsize = 6.5,
        handlelength = 2.0,
        columnspacing = 1.8,
        handletextpad = 0.55,
        borderaxespad = 0,
    )
    if legend is not None:
        for text in legend.get_texts():
            text.set_color(_MUTED_FOREGROUND)


def _monotone_curve(data: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    """Steffen interpolation used by Recharts/d3 for a ``monotone`` line."""
    if len(data) < 3:
        return ([step for step, _value in data], [value for _step, value in data])

    x = [step for step, _value in data]
    y = [value for _step, value in data]
    widths = [x[index + 1] - x[index] for index in range(len(x) - 1)]
    if any(width <= 0 for width in widths):
        return x, y
    slopes = [(y[index + 1] - y[index]) / widths[index] for index in range(len(widths))]

    tangents = [0.0] * len(x)
    for index in range(1, len(x) - 1):
        before, after = slopes[index - 1], slopes[index]
        before_width, after_width = widths[index - 1], widths[index]
        weighted = (
            before * after_width + after * before_width
        ) / (before_width + after_width)
        direction = (-1 if before < 0 else 1) + (-1 if after < 0 else 1)
        tangents[index] = direction * min(
            abs(before),
            abs(after),
            0.5 * abs(weighted),
        )
    tangents[0] = (3 * slopes[0] - tangents[1]) / 2
    tangents[-1] = (3 * slopes[-1] - tangents[-2]) / 2

    curve_x: list[float] = []
    curve_y: list[float] = []
    samples = 16
    for index, width in enumerate(widths):
        for sample in range(samples):
            t = sample / samples
            t2, t3 = t * t, t * t * t
            curve_x.append(x[index] + t * width)
            curve_y.append(
                (2 * t3 - 3 * t2 + 1) * y[index]
                + (t3 - 2 * t2 + t) * width * tangents[index]
                + (-2 * t3 + 3 * t2) * y[index + 1]
                + (t3 - t2) * width * tangents[index + 1]
            )
    curve_x.append(x[-1])
    curve_y.append(y[-1])
    return curve_x, curve_y


def _render_loss_card(fig, card, series: MetricSeries, x_domain):
    axes = _card_axes(fig, card, "Training Loss")
    data = _compress(_ema(series.get("loss", [])))
    values = [value for _step, value, _smoothed in data]
    smoothed_values = [smoothed for _step, _value, smoothed in data]
    y_domain = _domain(values + smoothed_values)
    _set_common_axes(axes, x_domain, y_domain)

    if data:
        steps = [step for step, _value, _smoothed in data]
        point = "o" if len(data) <= 1 else None
        axes.plot(
            steps,
            values,
            color = _LOSS,
            linewidth = 1.2,
            alpha = 0.35,
            marker = point,
            markersize = 3,
            label = "Loss",
            solid_capstyle = "round",
            solid_joinstyle = "round",
        )
        axes.plot(
            steps,
            smoothed_values,
            color = _SMOOTHED,
            linewidth = 2.2,
            marker = point,
            markersize = 3,
            label = "Smoothed",
            solid_capstyle = "round",
            solid_joinstyle = "round",
        )
        raw = series.get("loss", [])
        average = round(sum(value for _step, value in raw) / len(raw), 4)
        axes.axhline(average, color = _LOSS, linewidth = 0.8, alpha = 0.5, dashes = (4, 4))
        axes.annotate(
            f"avg {_format_metric(average)}",
            xy = (x_domain[1], average),
            xytext = (-3, 3),
            textcoords = "offset points",
            color = _LOSS,
            fontsize = 6.25,
            ha = "right",
            va = "bottom",
        )
        _legend(axes, columns = 2)
    else:
        _empty_chart_message(axes, "No training loss recorded")


def _render_line_card(
    fig,
    card,
    title: str,
    data: list[tuple[float, float]],
    color: str,
    label: str,
    x_domain: tuple[float, float],
    *,
    learning_rate: bool = False,
):
    axes = _card_axes(fig, card, title)
    data = _compress(data)
    y_domain = _domain(value for _step, value in data)
    _set_common_axes(axes, x_domain, y_domain)
    if learning_rate:
        y_ticks = axes.get_yticks()
        axes.set_yticks(y_ticks, labels = [_format_lr_axis(value) for value in y_ticks])

    if data:
        axes.plot(
            [step for step, _value in data],
            [value for _step, value in data],
            color = color,
            linewidth = 2,
            marker = "o" if len(data) <= 1 else None,
            markersize = 3,
            label = label,
            solid_capstyle = "round",
            solid_joinstyle = "round",
        )
        _legend(axes)
    else:
        _empty_chart_message(axes, f"No {title.lower()} recorded")


def _empty_chart_message(axes, message: str, *, detail: Optional[str] = None):
    axes.text(
        0.5,
        0.54,
        message,
        transform = axes.transAxes,
        color = _MUTED_FOREGROUND,
        fontsize = 8,
        fontweight = 500,
        ha = "center",
        va = "center",
    )
    if detail:
        axes.text(
            0.5,
            0.43,
            detail,
            transform = axes.transAxes,
            color = _MUTED_FOREGROUND,
            alpha = 0.65,
            fontsize = 6.5,
            ha = "center",
            va = "center",
        )


def _render_eval_card(fig, card, data: list[tuple[float, float]]):
    axes = _card_axes(fig, card, "Eval Loss")
    data = _compress(data)
    if data:
        x_domain = (data[0][0], data[-1][0])
        if x_domain[0] == x_domain[1]:
            x_domain = (x_domain[0], x_domain[0] + 1)
        _set_common_axes(axes, x_domain, _domain(value for _step, value in data))
        curve_x, curve_y = _monotone_curve(data)
        axes.plot(
            curve_x,
            curve_y,
            color = _EVAL_LOSS,
            linewidth = 2,
            label = "Eval Loss",
            solid_capstyle = "round",
            solid_joinstyle = "round",
        )
        axes.scatter(
            [step for step, _value in data],
            [value for _step, value in data],
            color = _EVAL_LOSS,
            edgecolors = "none",
            s = 8,
            zorder = 3,
        )
        _legend(axes)
        return

    placeholder = [(0, 2.8), (50, 2.4), (100, 2.0), (150, 1.7), (200, 1.5)]
    _set_common_axes(axes, (0, 200), _domain(value for _step, value in placeholder))
    axes.plot(
        [step for step, _value in placeholder],
        [value for _step, value in placeholder],
        color = _EVAL_LOSS,
        linewidth = 2,
        alpha = 0.12,
    )
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_alpha(0.16)
    _empty_chart_message(
        axes,
        "Evaluation not configured",
        detail = "Set eval dataset & eval_steps to track eval loss",
    )


def render_training_stats_png(series: MetricSeries) -> bytes:
    """Render the same default four-card view used by Studio Training History."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(
        figsize = (_IMAGE_WIDTH / _DPI, _IMAGE_HEIGHT / _DPI),
        dpi = _DPI,
        facecolor = _BACKGROUND,
    )
    FigureCanvasAgg(fig)

    margin = 28
    gap = 24
    card_width = (_IMAGE_WIDTH - margin * 2 - gap) // 2
    card_height = (_IMAGE_HEIGHT - margin * 2 - gap) // 2
    left = margin
    right = margin + card_width + gap
    bottom = margin
    top = margin + card_height + gap
    cards = (
        (left, top, card_width, card_height),
        (right, top, card_width, card_height),
        (left, bottom, card_width, card_height),
        (right, bottom, card_width, card_height),
    )

    x_domain = _visible_step_domain(series)
    _render_loss_card(fig, cards[0], series, x_domain)
    _render_line_card(
        fig,
        cards[1],
        "Gradient Norm",
        series.get("grad_norm", []),
        _GRAD_NORM,
        "Grad Norm",
        x_domain,
    )
    _render_line_card(
        fig,
        cards[2],
        "Learning Rate",
        series.get("learning_rate", []),
        _LEARNING_RATE,
        "LR",
        x_domain,
        learning_rate = True,
    )
    _render_eval_card(fig, cards[3], series.get("eval_loss", []))

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format = "png",
        dpi = _DPI,
        facecolor = _BACKGROUND,
        metadata = {"Software": "Unsloth Studio", "Title": "Training Stats"},
    )
    return buffer.getvalue()


def build_training_stats_png(checkpoint_path: Any) -> tuple[bytes, bool]:
    """Return PNG bytes and whether real metrics were available for the checkpoint."""
    series = load_training_metric_series(checkpoint_path)
    return render_training_stats_png(series), _has_metrics(series)


def upload_training_stats_png(
    checkpoint_path: Any,
    repo_id: str,
    hf_token: str,
    *,
    api = None,
):
    """Upload ``training_stats.png`` as the final artifact in a model export."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token = hf_token)

    png, has_metrics = build_training_stats_png(checkpoint_path)
    result = api.upload_file(
        path_or_fileobj = png,
        path_in_repo = TRAINING_STATS_FILENAME,
        repo_id = repo_id,
        repo_type = "model",
        commit_message = "Add Unsloth training stats",
    )
    if has_metrics:
        logger.info("Uploaded Training History chart to %s/%s", repo_id, TRAINING_STATS_FILENAME)
    else:
        logger.info(
            "Uploaded Training History placeholder to %s/%s; no metrics matched checkpoint %s",
            repo_id,
            TRAINING_STATS_FILENAME,
            checkpoint_path,
        )
    return result
