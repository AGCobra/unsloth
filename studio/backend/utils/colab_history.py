# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Drive-backed training-history persistence for hosted runtimes (Colab).

Design: single writer per file, merge on read. Each Studio session owns one
snapshot directory on the shared drive (``<history_root>/sessions/<session_id>/``)
and is the only writer of the files inside it. Cross-session sharing happens by
reading other sessions' snapshots and merging their training tables into the
live DB. There are no shared write targets and therefore no locks -- Drive's
FUSE mount does not provide atomic primitives across VMs, so any lock built on
it (e.g. mkdir mutexes) can be held by two machines at once.

Layout under ``history_root`` (a folder inside the mounted drive):

    sessions/<session_id>/studio.db   history snapshot (single writer: that session)
    sessions/<session_id>/meta.json   heartbeat + format stamp (same writer)
    master/studio.db                  legacy (old notebook scheme); read-only here
    studio.db                         older legacy; read-only here

Snapshots contain only the training-history tables (``training_runs`` +
``training_metrics``), extracted from the live WAL-mode DB through a read-only
connection and uploaded atomically (write ``.tmp`` sibling, then ``os.replace``).
A full-file copy would race the WAL and would also publish unrelated tables
(chat threads, settings) to the drive.

Merging is idempotent and never downgrades: a run row is applied only when the
incoming copy has made at least as much progress (terminal status beats
``interrupted`` beats ``running``; ties broken by ``final_step``). ``running``
rows from a session whose heartbeat is fresh are skipped (that session still
owns them); from a stale session they are imported as ``interrupted`` so a
crashed VM's partial run survives.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

SNAPSHOT_FORMAT = 1

RUN_COLUMNS = (
    "id", "status", "model_name", "dataset_name", "config_json", "started_at",
    "ended_at", "total_steps", "final_step", "final_loss", "output_dir",
    "error_message", "duration_seconds", "loss_sparkline", "display_name",
)
RUN_REQUIRED = {"id", "status", "model_name", "dataset_name", "config_json", "started_at"}
METRIC_COLUMNS = (
    "run_id", "step", "loss", "learning_rate", "grad_norm", "eval_loss",
    "epoch", "num_tokens", "elapsed_seconds",
)
METRIC_REQUIRED = {"run_id", "step"}

# Rank of a run's lifecycle progress; merges never move a run down this ladder.
_TERMINAL_STATUSES = ("completed", "failed", "stopped", "cancelled", "error")


def _status_rank(status) -> int:
    status = (status or "").strip().lower()
    if status in _TERMINAL_STATUSES:
        return 2
    if status == "interrupted":
        return 1
    return 0  # running / unknown


def _connect(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if not read_only:
        db_path.parent.mkdir(parents = True, exist_ok = True)
    # check_same_thread=False: HistorySync uses its cached read connection from
    # the caller's thread and later from the sync thread, never concurrently
    # (start() happens-after prepare(); stop() joins the thread first).
    conn = sqlite3.connect(str(db_path), timeout = 60, check_same_thread = False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    if read_only:
        # query_only (not a mode=ro URI): guarantees no writes yet stays
        # compatible with reading a live WAL database owned by Studio.
        conn.execute("PRAGMA query_only=ON")
    return conn


def ensure_history_schema(conn: sqlite3.Connection) -> None:
    """History subset of the Studio schema (see storage/studio_db.py)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS training_runs (
            id TEXT NOT NULL PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'running',
            model_name TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            config_json TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            total_steps INTEGER,
            final_step INTEGER,
            final_loss REAL,
            output_dir TEXT,
            error_message TEXT,
            duration_seconds REAL,
            loss_sparkline TEXT,
            display_name TEXT
        )
        """
    )
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(training_runs)")}
    if "display_name" not in existing:
        conn.execute("ALTER TABLE training_runs ADD COLUMN display_name TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS training_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
            step INTEGER NOT NULL,
            loss REAL,
            learning_rate REAL,
            grad_norm REAL,
            eval_loss REAL,
            epoch REAL,
            num_tokens INTEGER,
            elapsed_seconds REAL,
            UNIQUE(run_id, step)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON training_metrics(run_id)")


def _table_columns(conn: sqlite3.Connection, table: str) -> set:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _select_exprs(columns, existing) -> str:
    return ", ".join(c if c in existing else f"NULL AS {c}" for c in columns)


def _chunks(values, size = 500):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _same_path(left, right) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return str(left) == str(right)


def merge_history(src_path, dst_path, *, on_running: str = "skip") -> tuple:
    """Merge training history from ``src_path`` into ``dst_path``.

    on_running: what to do with source rows still marked ``running`` --
      "skip" (their session is alive and owns them), "interrupt" (their
      session is dead; import as status ``interrupted``), or "import" (as-is,
      for a session syncing its own live DB outward).

    Returns (runs_applied, metrics_applied). Idempotent; never downgrades a
    destination row (see _status_rank).
    """
    if on_running not in ("skip", "interrupt", "import"):
        raise ValueError(f"invalid on_running: {on_running!r}")
    src_path, dst_path = Path(src_path), Path(dst_path)
    if not src_path.exists() or _same_path(src_path, dst_path):
        return 0, 0
    src = _connect(src_path, read_only = True)
    dst = _connect(dst_path)
    try:
        run_existing = _table_columns(src, "training_runs")
        if not RUN_REQUIRED.issubset(run_existing):
            return 0, 0
        metric_existing = _table_columns(src, "training_metrics")
        ensure_history_schema(dst)
        dst.commit()

        rows = src.execute(
            f"SELECT {_select_exprs(RUN_COLUMNS, run_existing)} FROM training_runs"
        ).fetchall()
        candidates = []
        for row in rows:
            record = {c: row[c] for c in RUN_COLUMNS}
            if _status_rank(record["status"]) == 0:
                if on_running == "skip":
                    continue
                if on_running == "interrupt":
                    record["status"] = "interrupted"
                    if not record["error_message"]:
                        record["error_message"] = "Session ended while this run was in progress."
            candidates.append(record)

        dst.execute("BEGIN IMMEDIATE")
        applied_ids = []
        for record in candidates:
            existing = dst.execute(
                "SELECT status, final_step FROM training_runs WHERE id = ?", (record["id"],)
            ).fetchone()
            if existing is not None:
                new_key = (_status_rank(record["status"]), record["final_step"] or 0)
                old_key = (_status_rank(existing["status"]), existing["final_step"] or 0)
                if new_key < old_key:
                    continue
            columns = ", ".join(RUN_COLUMNS)
            placeholders = ", ".join("?" for _ in RUN_COLUMNS)
            update = ", ".join(f"{c} = excluded.{c}" for c in RUN_COLUMNS if c != "id")
            dst.execute(
                f"INSERT INTO training_runs ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT(id) DO UPDATE SET {update}",
                tuple(record[c] for c in RUN_COLUMNS),
            )
            applied_ids.append(record["id"])

        metric_total = 0
        if applied_ids and METRIC_REQUIRED.issubset(metric_existing):
            metric_select = _select_exprs(METRIC_COLUMNS, metric_existing)
            metric_columns = ", ".join(METRIC_COLUMNS)
            metric_placeholders = ", ".join("?" for _ in METRIC_COLUMNS)
            metric_sql = f"""
                INSERT INTO training_metrics ({metric_columns})
                VALUES ({metric_placeholders})
                ON CONFLICT(run_id, step) DO UPDATE SET
                    loss = COALESCE(excluded.loss, loss),
                    learning_rate = COALESCE(excluded.learning_rate, learning_rate),
                    grad_norm = COALESCE(excluded.grad_norm, grad_norm),
                    eval_loss = COALESCE(excluded.eval_loss, eval_loss),
                    epoch = COALESCE(excluded.epoch, epoch),
                    num_tokens = COALESCE(excluded.num_tokens, num_tokens),
                    elapsed_seconds = COALESCE(excluded.elapsed_seconds, elapsed_seconds)
            """
            for id_chunk in _chunks(applied_ids):
                qmarks = ",".join("?" for _ in id_chunk)
                metric_rows = src.execute(
                    f"SELECT {metric_select} FROM training_metrics WHERE run_id IN ({qmarks})",
                    tuple(id_chunk),
                ).fetchall()
                if metric_rows:
                    dst.executemany(
                        metric_sql,
                        [tuple(row[c] for c in METRIC_COLUMNS) for row in metric_rows],
                    )
                    metric_total += len(metric_rows)
        dst.commit()
        return len(applied_ids), metric_total
    except Exception:
        try:
            dst.rollback()
        except sqlite3.Error:
            pass
        raise
    finally:
        src.close()
        dst.close()


def stop_studio_if_running() -> None:
    """Best-effort clean shutdown of a Studio server started via run.py."""
    try:
        import run
    except Exception:
        return
    try:
        shutdown = getattr(run, "_graceful_shutdown", None)
        if callable(shutdown):
            shutdown(getattr(run, "_server", None))
        shutdown_event = getattr(run, "_shutdown_event", None)
        if shutdown_event is not None:
            shutdown_event.set()
    except Exception as exc:
        logger.warning("Could not stop Studio cleanly: %r", exc)


class HistorySync:
    """Per-session history persistence against a shared drive folder.

    Safe with any number of concurrent sessions (e.g. two Colab VMs on the
    same Google Drive): each instance writes only inside its own
    ``sessions/<session_id>/`` directory and merges everyone else's snapshots
    read-only.
    """

    def __init__(
        self,
        *,
        history_root,
        live_db,
        session_id: str,
        snapshot_seconds: float = 20.0,
        pull_seconds: float = 60.0,
        stale_seconds: float = 15 * 60,
        max_consecutive_failures: int = 3,
        gc_days: float = 0.0,
    ):
        if not str(session_id).strip():
            raise ValueError("session_id must be non-empty")
        self.history_root = Path(history_root)
        self.live_db = Path(live_db)
        self.session_id = str(session_id).strip()
        self.sessions_dir = self.history_root / "sessions"
        self.session_dir = self.sessions_dir / self.session_id
        self.snapshot_path = self.session_dir / "studio.db"
        self.meta_path = self.session_dir / "meta.json"
        self.snapshot_seconds = float(snapshot_seconds)
        self.pull_seconds = float(pull_seconds)
        self.stale_seconds = float(stale_seconds)
        self.max_consecutive_failures = int(max_consecutive_failures)
        self.gc_days = float(gc_days)
        self.failures = []
        self.fatal_failure = None
        self._stop = threading.Event()
        self._thread = None
        self._live_ro = None
        self._last_data_version = None
        self._pull_watermarks = {}
        self._consecutive_failures = 0

    # ── snapshot (the only writes this session makes on the drive) ──

    def _heartbeat(self) -> None:
        self.session_dir.mkdir(parents = True, exist_ok = True)
        payload = json.dumps({
            "format": SNAPSHOT_FORMAT,
            "session_id": self.session_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(payload)
        os.replace(tmp, self.meta_path)

    def _live_changed(self) -> bool:
        """PRAGMA data_version changes when another connection commits."""
        if not self.live_db.exists():
            return False
        if self._live_ro is None:
            self._live_ro = _connect(self.live_db, read_only = True)
        version = self._live_ro.execute("PRAGMA data_version").fetchone()[0]
        if version == self._last_data_version:
            return False
        self._last_data_version = version
        return True

    def snapshot(self, *, force: bool = False) -> bool:
        """Extract history tables from the live DB and publish atomically.

        Includes ``running`` rows: this file has a single writer (us), so a
        crash costs at most one snapshot interval of history.
        """
        self._heartbeat()
        changed = self._live_changed()  # always: keeps the version watermark current
        if not (force or changed):
            return False
        with tempfile.TemporaryDirectory(prefix = "unsloth-history-") as tmpdir:
            local = Path(tmpdir) / "studio.db"
            conn = _connect(local)
            try:
                ensure_history_schema(conn)
                conn.commit()
            finally:
                conn.close()
            merge_history(self.live_db, local, on_running = "import")
            drive_tmp = self.snapshot_path.with_suffix(".db.tmp")
            shutil.copyfile(local, drive_tmp)
            os.replace(drive_tmp, self.snapshot_path)
        return True

    # ── merge-on-read ──

    def _source_is_stale(self, session_dir: Path) -> bool:
        newest = 0.0
        meta = session_dir / "meta.json"
        try:
            payload = json.loads(meta.read_text())
            updated = datetime.fromisoformat(payload["updated_at"])
            newest = updated.timestamp()
        except (OSError, ValueError, KeyError, TypeError):
            pass
        for name in ("meta.json", "studio.db"):
            try:
                newest = max(newest, (session_dir / name).stat().st_mtime)
            except OSError:
                pass
        return (time.time() - newest) > self.stale_seconds

    def _merge_in(self, src: Path, *, on_running: str) -> tuple:
        """Merge a foreign snapshot into the live DB, tolerating files caught
        mid-upload (retry once, then skip until the next pass)."""
        for attempt in (0, 1):
            try:
                return merge_history(src, self.live_db, on_running = on_running)
            except sqlite3.Error as exc:
                if attempt == 1:
                    logger.warning("Skipping unreadable history source %s: %r", src, exc)
                    return 0, 0
                time.sleep(1)
        return 0, 0

    def _foreign_sessions(self):
        if not self.sessions_dir.is_dir():
            return
        for session_dir in sorted(self.sessions_dir.iterdir()):
            if not session_dir.is_dir() or session_dir.name == self.session_id:
                continue
            db = session_dir / "studio.db"
            if db.exists():
                yield session_dir, db

    def pull(self, *, ignore_watermarks: bool = False) -> tuple:
        """Merge every other session's snapshot into the live DB.

        A source is re-read only when its snapshot changed since the last pull,
        so finished sessions are absorbed once and local deletions are not
        resurrected by unchanged files.
        """
        runs_total = metrics_total = 0
        for session_dir, db in self._foreign_sessions():
            try:
                stat = db.stat()
                watermark = (stat.st_mtime_ns, stat.st_size)
            except OSError:
                continue
            if not ignore_watermarks and self._pull_watermarks.get(db) == watermark:
                continue
            on_running = "interrupt" if self._source_is_stale(session_dir) else "skip"
            runs, metrics = self._merge_in(db, on_running = on_running)
            self._pull_watermarks[db] = watermark
            runs_total += runs
            metrics_total += metrics
        return runs_total, metrics_total

    def _absorb_legacy(self) -> None:
        legacy = [self.history_root / "master" / "studio.db", self.history_root / "studio.db"]
        env_db = (os.environ.get("UNSLOTH_STUDIO_HISTORY_DB") or "").strip()
        if env_db:
            legacy.append(Path(env_db).expanduser())
        for candidate in legacy:
            if candidate.exists() and not _same_path(candidate, self.live_db):
                self._merge_in(candidate, on_running = "interrupt")

    def _gc_absorbed_sessions(self) -> None:
        """Optionally prune session dirs absorbed long ago (their history now
        lives in this session's snapshot). Never touches fresh sessions."""
        if self.gc_days <= 0:
            return
        cutoff = self.gc_days * 86400
        for session_dir, db in list(self._foreign_sessions()):
            try:
                age = time.time() - max(
                    db.stat().st_mtime, (session_dir / "meta.json").stat().st_mtime
                    if (session_dir / "meta.json").exists() else 0.0,
                )
            except OSError:
                continue
            if age > cutoff and db in self._pull_watermarks:
                shutil.rmtree(session_dir, ignore_errors = True)
                self._pull_watermarks.pop(db, None)

    def prepare(self) -> None:
        """Startup: seed the live DB with all history visible on the drive,
        then publish our first snapshot."""
        self.history_root.mkdir(parents = True, exist_ok = True)
        self.sessions_dir.mkdir(parents = True, exist_ok = True)
        conn = _connect(self.live_db)
        try:
            ensure_history_schema(conn)
            conn.commit()
        finally:
            conn.close()
        self._absorb_legacy()
        self.pull(ignore_watermarks = True)
        self._gc_absorbed_sessions()
        self.snapshot(force = True)

    # ── background loop ──

    def _tick(self, now: float, next_pull: float) -> float:
        if now >= next_pull:
            self.pull()
            next_pull = now + self.pull_seconds
        self.snapshot()
        return next_pull

    def _loop(self, on_fatal) -> None:
        next_pull = time.monotonic() + self.pull_seconds
        while not self._stop.wait(self.snapshot_seconds):
            try:
                next_pull = self._tick(time.monotonic(), next_pull)
                self._consecutive_failures = 0
            except BaseException as exc:
                self._consecutive_failures += 1
                self.failures.append(exc)
                logger.warning(
                    "History sync failure %d/%d: %r",
                    self._consecutive_failures, self.max_consecutive_failures, exc,
                )
                if self._consecutive_failures >= self.max_consecutive_failures:
                    logger.error("History persistence lost; invoking fatal handler.")
                    self.fatal_failure = exc
                    if on_fatal is not None:
                        on_fatal(exc)
                    return

    def start(self, on_fatal = None) -> None:
        if self._thread is not None:
            raise RuntimeError("HistorySync already started")
        self._stop.clear()
        self._thread = threading.Thread(
            target = self._loop, args = (on_fatal,), name = "unsloth-history-sync", daemon = True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the loop and publish a final snapshot (raises on failure so
        callers can surface lost persistence)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout = 15)
            self._thread = None
        try:
            self.snapshot(force = True)
        finally:
            if self._live_ro is not None:
                self._live_ro.close()
                self._live_ro = None
