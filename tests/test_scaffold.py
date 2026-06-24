"""Smoke tests for the project scaffold.

Confirms the package imports and config invariants hold. Module behavior is
tested in later steps as each module is implemented.
"""

from findr import config


def test_blend_weights_sum_to_one():
    assert abs(config.SEMANTIC_WEIGHT + config.KEYWORD_WEIGHT - 1.0) < 1e-9


def test_api_bound_locally():
    # Hard invariant: the engine never binds to a public interface.
    assert config.API_HOST == "127.0.0.1"


def test_embed_dim_is_768():
    assert config.EMBED_DIM == 768


def test_supported_extensions_lowercase_no_dot():
    for ext in config.SUPPORTED_EXTENSIONS:
        assert ext == ext.lower()
        assert not ext.startswith(".")
