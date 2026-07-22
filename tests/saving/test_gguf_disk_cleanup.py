"""CPU-only checks for temporary GGUF export cleanup.

The helper is loaded directly from save.py's AST so these tests do not import Unsloth's GPU
stack. This keeps the file-cleanup contract testable on the lightweight CI runners.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


SAVE_PY = Path(__file__).resolve().parents[2] / "unsloth" / "save.py"
SAVE_SRC = SAVE_PY.read_text(encoding="utf-8")
SAVE_TREE = ast.parse(SAVE_SRC, filename=str(SAVE_PY))


def _top_level_node(name):
    for node in SAVE_TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.Assign)):
            names = [
                target.id for target in getattr(node, "targets", ()) if isinstance(target, ast.Name)
            ]
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
            if name in names:
                return node
    raise AssertionError(f"{name!r} not found in {SAVE_PY.name}")


def _load_cleanup_helper():
    module = ast.Module(
        body=[
            _top_level_node("_TEMPORARY_MERGED_WEIGHT_PATTERNS"),
            _top_level_node("_remove_temporary_merged_weights"),
        ],
        type_ignores=[],
    )
    namespace = {"Path": Path}
    exec(compile(ast.fix_missing_locations(module), str(SAVE_PY), "exec"), namespace)
    return namespace["_remove_temporary_merged_weights"]


def _calls(function_name, called_name):
    function = _top_level_node(function_name)
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == called_name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == called_name)
        )
    ]


def test_cleanup_removes_only_merged_weights_and_preserves_gguf_bytes(tmp_path):
    remove_weights = _load_cleanup_helper()
    removable = {
        "model.safetensors": b"single safe shard",
        "model-00001-of-00002.safetensors": b"first safe shard",
        "model.test-00002-of-00002.safetensors": b"variant safe shard",
        "model.safetensors.index.json": b"safe index",
        "pytorch_model.bin": b"single bin shard",
        "pytorch_model.test-00001-of-00002.bin": b"variant bin shard",
        "pytorch_model.bin.index.json": b"bin index",
    }
    preserved = {
        "config.json": b"config",
        "tokenizer.json": b"tokenizer",
        "adapter_model.safetensors": b"adapter",
        "model.Q4_K_M.gguf": b"the exact gguf bytes to upload",
    }
    for name, contents in (removable | preserved).items():
        (tmp_path / name).write_bytes(contents)

    gguf_path = tmp_path / "model.Q4_K_M.gguf"
    gguf_hash_before = hashlib.sha256(gguf_path.read_bytes()).digest()

    removed_files, removed_bytes = remove_weights(tmp_path)

    assert {Path(path).name for path in removed_files} == set(removable)
    assert removed_bytes == sum(len(contents) for contents in removable.values())
    assert all(not (tmp_path / name).exists() for name in removable)
    assert all((tmp_path / name).read_bytes() == contents for name, contents in preserved.items())
    assert hashlib.sha256(gguf_path.read_bytes()).digest() == gguf_hash_before


def test_cleanup_runs_only_after_initial_gguf_conversion_and_before_quantization():
    conversion = _calls("save_to_gguf", "convert_to_gguf")
    cleanup = _calls("save_to_gguf", "_remove_temporary_merged_weights")
    quantization = _calls("save_to_gguf", "quantize_gguf")

    assert len(conversion) == len(cleanup) == len(quantization) == 1
    assert conversion[0].lineno < cleanup[0].lineno < quantization[0].lineno


def test_hub_export_enables_cleanup_only_for_owned_temp_directory():
    calls = _calls("unsloth_push_to_hub_gguf", "unsloth_save_pretrained_gguf")
    assert len(calls) == 1
    keyword = next(
        kw for kw in calls[0].keywords if kw.arg == "_delete_intermediate_merged_weights"
    )
    assert isinstance(keyword.value, ast.Name) and keyword.value.id == "cleanup_temp"

    saver_source = ast.get_source_segment(SAVE_SRC, _top_level_node("unsloth_save_pretrained_gguf"))
    assert (
        "os.path.realpath(os.fspath(save_directory)) == temporary_merged_directory" in saver_source
    )


def test_temporary_gguf_is_deleted_only_after_synchronous_upload():
    uploads = _calls("unsloth_push_to_hub_gguf", "upload_file")
    unlinks = _calls("unsloth_push_to_hub_gguf", "unlink")

    assert uploads and len(unlinks) == 1
    assert min(call.lineno for call in uploads) < unlinks[0].lineno
