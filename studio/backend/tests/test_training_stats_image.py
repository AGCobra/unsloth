# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import sqlite3
import struct
import sys
import types
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_MODULE_PATH = _BACKEND_DIR / "core" / "export" / "training_stats_image.py"
_EXPORT_PATH = _BACKEND_DIR / "core" / "export" / "export.py"


@pytest.fixture
def stats_module(monkeypatch):
    loggers = types.ModuleType("loggers")
    loggers.get_logger = logging.getLogger
    storage = types.ModuleType("storage")
    studio_db = types.ModuleType("storage.studio_db")
    studio_db.get_connection = lambda: (_ for _ in ()).throw(RuntimeError("no database"))
    monkeypatch.setitem(sys.modules, "loggers", loggers)
    monkeypatch.setitem(sys.modules, "storage", storage)
    monkeypatch.setitem(sys.modules, "storage.studio_db", studio_db)

    spec = importlib.util.spec_from_file_location("test_training_stats_image_module", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def test_metric_series_uses_newest_history_run_for_checkpoint(stats_module, tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs" / "run"
    checkpoint = output_dir / "checkpoint-20"
    checkpoint.mkdir(parents = True)
    db_path = tmp_path / "studio.db"
    conn = _connection(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE training_runs (
                id TEXT PRIMARY KEY,
                output_dir TEXT,
                started_at TEXT
            );
            CREATE TABLE training_metrics (
                run_id TEXT,
                step INTEGER,
                loss REAL,
                learning_rate REAL,
                grad_norm REAL,
                eval_loss REAL
            );
            """
        )
        conn.executemany(
            "INSERT INTO training_runs (id, output_dir, started_at) VALUES (?, ?, ?)",
            [
                ("old", str(output_dir), "2026-01-01T00:00:00Z"),
                ("new", str(output_dir), "2026-01-02T00:00:00Z"),
            ],
        )
        conn.executemany(
            """
            INSERT INTO training_metrics
                (run_id, step, loss, learning_rate, grad_norm, eval_loss)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ("old", 1, 9.0, 9e-4, 9.0, None),
                ("new", 1, 2.0, 2e-4, 0.8, None),
                ("new", 2, 1.5, 1e-4, 0.7, 1.7),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(stats_module, "get_connection", lambda: _connection(db_path))

    series = stats_module.load_training_metric_series(checkpoint)

    assert series == {
        "loss": [(1.0, 2.0), (2.0, 1.5)],
        "learning_rate": [(1.0, 2e-4), (2.0, 1e-4)],
        "grad_norm": [(1.0, 0.8), (2.0, 0.7)],
        "eval_loss": [(2.0, 1.7)],
    }


def test_metric_series_falls_back_to_latest_trainer_state(stats_module, tmp_path):
    output_dir = tmp_path / "run"
    older = output_dir / "checkpoint-3"
    latest = output_dir / "checkpoint-11"
    older.mkdir(parents = True)
    latest.mkdir()
    (older / "trainer_state.json").write_text(
        json.dumps({"log_history": [{"step": 3, "loss": 8.0}]}), encoding = "utf-8"
    )
    (latest / "trainer_state.json").write_text(
        json.dumps(
            {
                "log_history": [
                    {"step": 10, "loss": 1.25, "learning_rate": 2e-4, "grad_norm": 0.9},
                    {"step": 10, "eval_loss": 1.4},
                    {"step": 11, "loss": 1.1, "learning_rate": 1e-4, "grad_norm": 0.8},
                ]
            }
        ),
        encoding = "utf-8",
    )

    series = stats_module.load_training_metric_series(output_dir)

    assert series["loss"] == [(10.0, 1.25), (11.0, 1.1)]
    assert series["learning_rate"] == [(10.0, 2e-4), (11.0, 1e-4)]
    assert series["grad_norm"] == [(10.0, 0.9), (11.0, 0.8)]
    assert series["eval_loss"] == [(10.0, 1.4)]


def test_renderer_produces_studio_sized_png_with_page_defaults(stats_module):
    png = stats_module.render_training_stats_png(
        {
            "loss": [(1, 2.0), (2, 1.5), (3, 1.2)],
            "learning_rate": [(1, 2e-4), (2, 1e-4), (3, 5e-5)],
            "grad_norm": [(1, 0.9), (2, 0.8), (3, 0.7)],
            "eval_loss": [(1, 1.8), (3, 1.3)],
        }
    )

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", png[16:24]) == (1440, 760)
    assert len(png) > 25_000


def test_upload_always_adds_named_png_even_without_metrics(stats_module, monkeypatch):
    expected_png = b"\x89PNG\r\n\x1a\nplaceholder"
    monkeypatch.setattr(stats_module, "build_training_stats_png", lambda _path: (expected_png, False))

    class FakeApi:
        def __init__(self):
            self.kwargs = None

        def upload_file(self, **kwargs):
            self.kwargs = kwargs
            return "commit"

    api = FakeApi()

    result = stats_module.upload_training_stats_png(
        "/models/external",
        "user/model",
        "secret-token",
        api = api,
    )

    assert result == "commit"
    assert api.kwargs == {
        "path_or_fileobj": expected_png,
        "path_in_repo": "training_stats.png",
        "repo_id": "user/model",
        "repo_type": "model",
        "commit_message": "Add Unsloth training stats",
    }


@pytest.mark.parametrize(
    "function_name",
    ["export_merged_model", "export_base_model", "export_gguf", "export_lora_adapter"],
)
def test_every_studio_model_hub_export_attaches_training_stats(function_name):
    tree = ast.parse(_EXPORT_PATH.read_text(encoding = "utf-8"))
    function = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    hub_branch = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "push_to_hub"
    )
    calls = [
        node
        for node in ast.walk(hub_branch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_upload_training_stats"
    ]
    assert len(calls) == 1
