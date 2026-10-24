"""CPU-only projection coverage; execute the benchmark's actual shape helper.

The helper is pure Python. Loading just its AST avoids importing CUDA/PyTorch
or loading an extension merely to validate architecture-derived dimensions.
"""
import ast
import json
import os
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "benchmarks/bench_qwen3_gemm.py"
DEFAULT_MODEL_CONFIG = next((path for parent in SOURCE.resolve().parents
    if (path := parent / "myvllm" / "models" / "Qwen3-0.6B" / "config.json").is_file()), None)


@pytest.fixture(scope="module")
def shapes_for_config():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    helpers = [node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == "projection_shapes"]
    assert len(helpers) == 1, "benchmark must expose one projection_shapes helper"
    namespace = {}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["projection_shapes"]


def _assert_projection_contract(config, shapes):
    # Derive tensor widths from the architecture rather than assuming that
    # query width equals hidden_size (it does not in Qwen3-0.6B).
    hidden = config["hidden_size"]
    head_dim = config.get("head_dim", hidden // config["num_attention_heads"])
    query_width = config["num_attention_heads"] * head_dim
    kv_width = config["num_key_value_heads"] * head_dim
    mlp_width = config["intermediate_size"]
    assert set(shapes) == {
        "q_proj", "kv_proj", "o_proj", "qkv_fused",
        "gate_up_single", "gate_up_fused", "down_proj",
    }
    assert shapes["q_proj"] == (query_width, hidden)
    assert shapes["kv_proj"] == (kv_width, hidden)
    assert shapes["o_proj"] == (hidden, query_width)
    assert shapes["qkv_fused"] == (query_width + 2 * kv_width, hidden)
    assert shapes["gate_up_single"] == (mlp_width, hidden)
    assert shapes["gate_up_fused"] == (2 * mlp_width, hidden)
    assert shapes["down_proj"] == (hidden, mlp_width)
    # These relations validate how the projection operators compose.
    assert shapes["q_proj"][0] == shapes["o_proj"][1]
    assert shapes["o_proj"][0] == shapes["q_proj"][1]
    assert shapes["gate_up_single"] == tuple(reversed(shapes["down_proj"]))


def test_actual_qwen3_config_covers_attention_output(shapes_for_config):
    configured = os.environ.get("QWEN3_MODEL_CONFIG", DEFAULT_MODEL_CONFIG)
    if configured is None:
        pytest.skip("optional local Qwen3 model config not found")
    path = Path(configured)
    config = json.loads(path.read_text(encoding="utf-8"))
    shapes = shapes_for_config(config)
    _assert_projection_contract(config, shapes)
    # The shipped model makes this regression observable; using (hidden,hidden)
    # for o_proj would miss the real 1024x2048 attention output projection.
    if path == DEFAULT_MODEL_CONFIG:
        assert shapes["o_proj"] == (1024, 2048)
        assert config["hidden_size"] != config["num_attention_heads"] * config["head_dim"]


@pytest.mark.parametrize("config", [
    {"hidden_size": 768, "intermediate_size": 2304,
     "num_attention_heads": 12, "num_key_value_heads": 3, "head_dim": 128},
    {"hidden_size": 960, "intermediate_size": 2560,
     "num_attention_heads": 12, "num_key_value_heads": 4},
    {"hidden_size": 256, "intermediate_size": 704,
     "num_attention_heads": 4, "num_key_value_heads": 4, "head_dim": 64},
])
def test_all_projection_shapes_follow_configuration(shapes_for_config, config):
    _assert_projection_contract(config, shapes_for_config(config))


def test_explicit_head_dim_changes_query_and_output_width_together(shapes_for_config):
    config = {"hidden_size": 768, "intermediate_size": 2048,
              "num_attention_heads": 12, "num_key_value_heads": 2}
    inferred = shapes_for_config(config)
    explicit = shapes_for_config({**config, "head_dim": 128})
    assert inferred["q_proj"] == (768, 768)
    assert explicit["q_proj"] == (1536, 768)
    assert explicit["o_proj"] == (768, 1536)
    assert explicit["qkv_fused"][0] > inferred["qkv_fused"][0]
    for unaffected in ("gate_up_single", "gate_up_fused", "down_proj"):
        assert explicit[unaffected] == inferred[unaffected]
