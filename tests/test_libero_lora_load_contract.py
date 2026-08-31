from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _only_lora_load(path: Path) -> ast.Call:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "load_lora_checkpoint"
    ]
    assert len(calls) == 1
    return calls[0]


def _keyword_map(call: ast.Call) -> dict[str, ast.expr]:
    assert all(keyword.arg is not None for keyword in call.keywords)
    return {str(keyword.arg): keyword.value for keyword in call.keywords}


def _same_expression(observed: ast.expr, expected: str) -> bool:
    parsed = ast.parse(expected, mode="eval")
    return ast.dump(observed, include_attributes=False) == ast.dump(parsed.body, include_attributes=False)


def test_libero_resume_strictly_validates_decoder_lora_schema() -> None:
    call = _only_lora_load(PROJECT_ROOT / "scripts/train_libero.py")
    keywords = _keyword_map(call)

    assert ast.literal_eval(keywords["is_trainable"]) is True
    assert ast.literal_eval(keywords["validate_decoder_contract"]) is True
    assert _same_expression(keywords["expected_rank"], "int(config['lora']['rank'])")


def test_libero_server_strictly_validates_decoder_lora_schema() -> None:
    call = _only_lora_load(PROJECT_ROOT / "scripts/serve_libero_policy.py")
    keywords = _keyword_map(call)

    assert ast.literal_eval(keywords["is_trainable"]) is False
    assert ast.literal_eval(keywords["validate_decoder_contract"]) is True
    assert _same_expression(keywords["expected_rank"], "int(resolved_config['lora']['rank'])")
