# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Single-writer-per-file training-history sync for hosted runtimes.

Pins the concurrency contract that lets two Colab VMs share one Drive history
folder: each session writes only its own ``sessions/<id>/`` snapshot, merges
everyone else's read-only, never downgrades run progress, and rescues stale
``running`` rows as ``interrupted``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Dependency-light module: load directly so these tests run without the studio venv.
_CH_PATH = Path(_BACKEND_DIR) / "utils" / "colab_history.py"
_spec = importlib.util.spec_from_file_location("_colab_history_test", _CH_PATH)
ch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ch)


def _run_row(run_id, status = "completed", final_step = 10, **overrides):
    row = {
        "id": run_id,
        "status": status,
        "model_name": "unsloth/gemma-4-12B-it",
        "dataset_name": "ds",
        "config_json": "{}",
        "started_at": "2026-07-01T00:00:00Z",
        "ended_at": None,
        "total_steps": 100,
        "final_step": final_step,
        "final_loss": 1.0,
        "output_dir": None,
        "error_message": None,
        "duration_seconds": None,
        "loss_sparkline": None,
        "display_name": None,
    }
    row.update(overrides)
    return row


def _make_db(path, runs = (), metrics = ()):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    ch.ensure_history_schema(conn)
    for run in runs:
        cols = ", ".join(run)
        conn.execute(
            f"INSERT INTO training_runs ({cols}) VALUES ({', '.join('?' for _ in run)})",
            tuple(run.values()),
        )
    for metric in metrics:
        cols = ", ".join(metric)
        conn.execute(
            f"INSERT INTO training_metrics ({cols}) VALUES ({', '.join('?' for _ in metric)})",
            tuple(metric.values()),
        )
    conn.commit()
    conn.close()
    return path


def _runs(path):
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        return {
            row["id"]: dict(row)
            for row in conn.execute("SELECT * FROM training_runs").fetchall()
        }
    finally:
        conn.close()


def _metrics(path, run_id):
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        return {
            row["step"]: dict(row)
            for row in conn.execute(
                "SELECT * FROM training_metrics WHERE run_id = ?", (run_id,)
            ).fetchall()
        }
    finally:
        conn.close()


def _age(path, seconds):
    old = time.time() - seconds
    os.utime(path, (old, old))


# ── merge_history ────────────────────────────────────────────────────


def test_merge_copies_completed_runs_and_metrics(tmp_path):
    src = _make_db(
        tmp_path / "src.db",
        runs = [_run_row("r1")],
        metrics = [{"run_id": "r1", "step": 1, "loss": 2.0}],
    )
    dst = _make_db(tmp_path / "dst.db")
    runs, metrics = ch.merge_history(src, dst)
    assert (runs, metrics) == (1, 1)
    assert _runs(dst)["r1"]["status"] == "completed"
    assert _metrics(dst, "r1")[1]["loss"] == 2.0
    # Idempotent.
    ch.merge_history(src, dst)
    assert len(_runs(dst)) == 1


def test_merge_never_downgrades_a_completed_run(tmp_path):
    dst = _make_db(tmp_path / "dst.db", runs = [_run_row("r1", "completed", final_step = 100)])
    src = _make_db(tmp_path / "src.db", runs = [_run_row("r1", "running", final_step = 50)])
    runs, _ = ch.merge_history(src, dst, on_running = "import")
    assert runs == 0
    assert _runs(dst)["r1"]["status"] == "completed"
    assert _runs(dst)["r1"]["final_step"] == 100
    # interrupted (rank 1) also cannot clobber a terminal row.
    src2 = _make_db(tmp_path / "src2.db", runs = [_run_row("r1", "interrupted", final_step = 999)])
    runs, _ = ch.merge_history(src2, dst)
    assert runs == 0
    assert _runs(dst)["r1"]["status"] == "completed"


def test_merge_upgrades_running_to_completed(tmp_path):
    dst = _make_db(tmp_path / "dst.db", runs = [_run_row("r1", "running", final_step = 10)])
    src = _make_db(tmp_path / "src.db", runs = [_run_row("r1", "completed", final_step = 100)])
    runs, _ = ch.merge_history(src, dst)
    assert runs == 1
    assert _runs(dst)["r1"]["status"] == "completed"
    assert _runs(dst)["r1"]["final_step"] == 100


def test_merge_on_running_modes(tmp_path):
    src = _make_db(tmp_path / "src.db", runs = [_run_row("r1", "running")])
    skip_dst = _make_db(tmp_path / "skip.db")
    assert ch.merge_history(src, skip_dst, on_running = "skip") == (0, 0)
    assert _runs(skip_dst) == {}

    interrupt_dst = _make_db(tmp_path / "interrupt.db")
    runs, _ = ch.merge_history(src, interrupt_dst, on_running = "interrupt")
    assert runs == 1
    rescued = _runs(interrupt_dst)["r1"]
    assert rescued["status"] == "interrupted"
    assert "in progress" in rescued["error_message"]

    import_dst = _make_db(tmp_path / "import.db")
    ch.merge_history(src, import_dst, on_running = "import")
    assert _runs(import_dst)["r1"]["status"] == "running"

    with pytest.raises(ValueError):
        ch.merge_history(src, skip_dst, on_running = "bogus")


def test_merge_tolerates_older_schema_without_optional_columns(tmp_path):
    src = tmp_path / "old.db"
    conn = sqlite3.connect(str(src))
    conn.execute(
        """
        CREATE TABLE training_runs (
            id TEXT NOT NULL PRIMARY KEY,
            status TEXT NOT NULL,
            model_name TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            config_json TEXT NOT NULL,
            started_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO training_runs VALUES ('r1', 'completed', 'm', 'd', '{}', 't')"
    )
    conn.commit()
    conn.close()
    dst = _make_db(tmp_path / "dst.db")
    runs, metrics = ch.merge_history(src, dst)
    assert (runs, metrics) == (1, 0)
    assert _runs(dst)["r1"]["display_name"] is None


def test_merge_metrics_coalesce_fills_gaps_without_nulling(tmp_path):
    dst = _make_db(
        tmp_path / "dst.db",
        runs = [_run_row("r1", "running", final_step = 1)],
        metrics = [{"run_id": "r1", "step": 1, "loss": 2.0, "eval_loss": None}],
    )
    src = _make_db(
        tmp_path / "src.db",
        runs = [_run_row("r1", "completed", final_step = 2)],
        metrics = [{"run_id": "r1", "step": 1, "loss": None, "eval_loss": 3.0}],
    )
    ch.merge_history(src, dst)
    merged = _metrics(dst, "r1")[1]
    assert merged["loss"] == 2.0  # not nulled by the src row
    assert merged["eval_loss"] == 3.0  # filled from the src row


def test_merge_missing_source_is_a_noop(tmp_path):
    dst = _make_db(tmp_path / "dst.db")
    assert ch.merge_history(tmp_path / "absent.db", dst) == (0, 0)


# ── HistorySync: snapshot ────────────────────────────────────────────


def _sync(tmp_path, **kw):
    live = _make_db(tmp_path / "runtime" / "live.db")
    kw.setdefault("history_root", tmp_path / "drive")
    kw.setdefault("live_db", live)
    kw.setdefault("session_id", "sess-self")
    return ch.HistorySync(**kw)


def test_snapshot_publishes_history_and_heartbeat(tmp_path):
    sync = _sync(tmp_path)
    _make_db(sync.live_db, runs = [_run_row("r1", "running", final_step = 5)])
    assert sync.snapshot(force = True) is True
    assert sync.snapshot_path.exists()
    # Running rows are included: our own file, single writer, crash-durable.
    assert _runs(sync.snapshot_path)["r1"]["status"] == "running"
    meta = json.loads(sync.meta_path.read_text())
    assert meta["session_id"] == "sess-self"
    assert meta["format"] == ch.SNAPSHOT_FORMAT
    assert not list(sync.session_dir.glob("*.tmp"))
    sync.stop()


def test_snapshot_skips_when_live_db_unchanged(tmp_path):
    sync = _sync(tmp_path)
    assert sync.snapshot(force = True) is True
    assert sync.snapshot() is False  # no writes since
    conn = sqlite3.connect(str(sync.live_db))
    run = _run_row("r2")
    conn.execute(
        f"INSERT INTO training_runs ({', '.join(run)}) VALUES ({', '.join('?' for _ in run)})",
        tuple(run.values()),
    )
    conn.commit()
    conn.close()
    assert sync.snapshot() is True
    assert "r2" in _runs(sync.snapshot_path)
    sync.stop()


def test_snapshot_captures_wal_content(tmp_path):
    sync = _sync(tmp_path)
    conn = sqlite3.connect(str(sync.live_db))
    conn.execute("PRAGMA journal_mode=WAL")
    run = _run_row("r-wal")
    conn.execute(
        f"INSERT INTO training_runs ({', '.join(run)}) VALUES ({', '.join('?' for _ in run)})",
        tuple(run.values()),
    )
    conn.commit()  # committed to the -wal, not yet checkpointed into the db file
    assert sync.snapshot(force = True) is True
    assert "r-wal" in _runs(sync.snapshot_path)
    conn.close()
    sync.stop()


# ── HistorySync: prepare / pull ──────────────────────────────────────


def test_prepare_absorbs_legacy_master_and_sessions(tmp_path):
    sync = _sync(tmp_path)
    _make_db(tmp_path / "drive" / "master" / "studio.db", runs = [_run_row("legacy-master")])
    _make_db(tmp_path / "drive" / "studio.db", runs = [_run_row("legacy-root")])
    _make_db(
        tmp_path / "drive" / "sessions" / "sess-old" / "studio.db",
        runs = [_run_row("old-completed")],
    )
    sync.prepare()
    live = _runs(sync.live_db)
    assert {"legacy-master", "legacy-root", "old-completed"} <= set(live)
    # First snapshot published, so the newest session file is a superset.
    assert {"legacy-master", "legacy-root", "old-completed"} <= set(_runs(sync.snapshot_path))
    sync.stop()


def test_stale_running_run_is_rescued_as_interrupted(tmp_path):
    sync = _sync(tmp_path, stale_seconds = 60)
    dead = tmp_path / "drive" / "sessions" / "sess-dead"
    _make_db(dead / "studio.db", runs = [_run_row("crashed", "running", final_step = 7)])
    _age(dead / "studio.db", 3600)
    sync.prepare()
    rescued = _runs(sync.live_db)["crashed"]
    assert rescued["status"] == "interrupted"
    assert rescued["final_step"] == 7
    sync.stop()


def test_fresh_running_run_is_left_to_its_owner(tmp_path):
    sync = _sync(tmp_path, stale_seconds = 3600)
    live_peer = tmp_path / "drive" / "sessions" / "sess-live"
    _make_db(
        live_peer / "studio.db",
        runs = [_run_row("peer-running", "running"), _run_row("peer-done", "completed")],
    )
    sync.prepare()
    live = _runs(sync.live_db)
    assert "peer-running" not in live
    assert "peer-done" in live
    sync.stop()


def test_pull_skips_unchanged_sources_so_local_deletes_stick(tmp_path):
    sync = _sync(tmp_path)
    peer_db = tmp_path / "drive" / "sessions" / "sess-peer" / "studio.db"
    _make_db(peer_db, runs = [_run_row("peer-1")])
    sync.prepare()
    assert "peer-1" in _runs(sync.live_db)

    conn = sqlite3.connect(str(sync.live_db))
    conn.execute("DELETE FROM training_runs WHERE id = 'peer-1'")
    conn.commit()
    conn.close()
    sync.pull()
    assert "peer-1" not in _runs(sync.live_db)  # unchanged source not re-read

    _age(peer_db, -2)  # source changed -> re-merged on the next pull
    sync.pull()
    assert "peer-1" in _runs(sync.live_db)
    sync.stop()


def test_own_session_dir_is_never_pulled(tmp_path):
    sync = _sync(tmp_path)
    _make_db(
        tmp_path / "drive" / "sessions" / "sess-self" / "studio.db",
        runs = [_run_row("self-echo")],
    )
    sync.prepare()
    # prepare() replaced our own snapshot rather than merging it back in.
    assert "self-echo" not in _runs(sync.live_db)
    sync.stop()


# ── HistorySync: gc + loop ───────────────────────────────────────────


def test_gc_disabled_by_default_and_prunes_only_old_absorbed_sessions(tmp_path):
    sync = _sync(tmp_path)
    old = tmp_path / "drive" / "sessions" / "sess-ancient"
    _make_db(old / "studio.db", runs = [_run_row("ancient")])
    _age(old / "studio.db", 90 * 86400)
    sync.prepare()
    assert old.exists()  # gc_days = 0 -> off
    sync.stop()

    sync2 = _sync(tmp_path, session_id = "sess-gc", gc_days = 30)
    fresh = tmp_path / "drive" / "sessions" / "sess-fresh"
    _make_db(fresh / "studio.db", runs = [_run_row("fresh-run")])
    sync2.prepare()
    assert not old.exists()  # absorbed and older than 30d -> pruned
    assert fresh.exists()
    assert "ancient" in _runs(sync2.live_db)  # history survived the prune
    sync2.stop()


def test_background_loop_snapshots_and_final_stop_snapshot(tmp_path):
    sync = _sync(tmp_path, snapshot_seconds = 0.05, pull_seconds = 0.1)
    sync.prepare()
    conn = sqlite3.connect(str(sync.live_db))
    run = _run_row("looped")
    conn.execute(
        f"INSERT INTO training_runs ({', '.join(run)}) VALUES ({', '.join('?' for _ in run)})",
        tuple(run.values()),
    )
    conn.commit()
    conn.close()
    sync.start()
    deadline = time.time() + 5
    while time.time() < deadline and "looped" not in _runs(sync.snapshot_path):
        time.sleep(0.05)
    sync.stop()
    assert "looped" in _runs(sync.snapshot_path)
    assert sync.failures == []


def test_two_concurrent_sessions_share_history_without_shared_writes(tmp_path):
    """The two-Colab scenario: both sessions on one Drive history folder."""
    drive = tmp_path / "drive"
    a = ch.HistorySync(
        history_root = drive, session_id = "colab-a",
        live_db = _make_db(tmp_path / "vm-a" / "studio.db"),
    )
    b = ch.HistorySync(
        history_root = drive, session_id = "colab-b",
        live_db = _make_db(tmp_path / "vm-b" / "studio.db"),
    )
    a.prepare()
    b.prepare()

    def _insert(db, run):
        conn = sqlite3.connect(str(db))
        conn.execute(
            f"INSERT INTO training_runs ({', '.join(run)}) VALUES ({', '.join('?' for _ in run)})",
            tuple(run.values()),
        )
        conn.commit()
        conn.close()

    _insert(a.live_db, _run_row("a-done", "completed"))
    _insert(a.live_db, _run_row("a-active", "running"))
    _insert(b.live_db, _run_row("b-done", "completed"))
    a.snapshot(force = True)
    b.snapshot(force = True)

    a.pull(ignore_watermarks = True)
    b.pull(ignore_watermarks = True)

    # Completed runs flow both ways; live "running" rows stay with their owner.
    assert "b-done" in _runs(a.live_db)
    assert "a-done" in _runs(b.live_db)
    assert "a-active" not in _runs(b.live_db)

    # Neither session ever wrote outside its own directory: no shared file.
    a_files = {p.name for p in a.session_dir.iterdir()}
    b_files = {p.name for p in b.session_dir.iterdir()}
    assert a.session_dir != b.session_dir
    assert "studio.db" in a_files and "studio.db" in b_files

    # A third session started later sees everything, including A's crashed run.
    # Simulate A's VM dying an hour ago: mtimes AND the heartbeat payload age.
    from datetime import datetime, timedelta, timezone

    meta = json.loads((a.session_dir / "meta.json").read_text())
    meta["updated_at"] = (datetime.now(timezone.utc) - timedelta(hours = 1)).isoformat()
    (a.session_dir / "meta.json").write_text(json.dumps(meta))
    _age(a.session_dir / "studio.db", 3600)
    _age(a.session_dir / "meta.json", 3600)
    c = ch.HistorySync(
        history_root = drive, session_id = "colab-c", stale_seconds = 60,
        live_db = _make_db(tmp_path / "vm-c" / "studio.db"),
    )
    c.prepare()
    live_c = _runs(c.live_db)
    assert {"a-done", "b-done", "a-active"} <= set(live_c)
    assert live_c["a-active"]["status"] == "interrupted"
    for s in (a, b, c):
        s.stop()


def test_fatal_handler_fires_after_consecutive_failures(tmp_path):
    sync = _sync(tmp_path, snapshot_seconds = 0.01, max_consecutive_failures = 2)
    sync.prepare()
    fired = []
    # Break snapshotting: replace the session dir path with an un-creatable one.
    sync.session_dir = sync.live_db / "not-a-dir" / "x"  # parent is a file
    sync.meta_path = sync.session_dir / "meta.json"
    sync.snapshot_path = sync.session_dir / "studio.db"
    sync.start(on_fatal = fired.append)
    deadline = time.time() + 5
    while time.time() < deadline and not fired:
        time.sleep(0.02)
    sync._stop.set()
    assert fired, "fatal handler did not fire"
    assert len(sync.failures) >= 2
