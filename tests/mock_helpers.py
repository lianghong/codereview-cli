"""Shared helpers for the tests that mock `CodeAnalyzer` wholesale.

`run_review` reads the *accrued* cost off the analyzer rather than recomputing
`tokens * rate`, because on a tiered entry the rate depends on one request's
input size (→ `docs/providers.md`). A `Mock()` analyzer returns a `Mock` from
`estimate_cost()`, which is not subscriptable, so every such fixture has to
supply a real dict.

`wire_mock_cost` derives that dict from the numbers the fixture already set, so
a mocked run reports arithmetic consistent with its own token counts instead of
an invented constant — a fixture that lies about cost is how a wrong cost
figure survives a green suite.
"""

from typing import Any


def wire_mock_cost(mock_analyzer: Any, mock_provider: Any) -> dict[str, float]:
    """Give ``mock_analyzer.estimate_cost()`` a real dict matching its tokens.

    Reads ``total_input_tokens``/``total_output_tokens`` and the rates from
    ``mock_provider.get_pricing()``, falling back to 0.0 for anything a
    fixture left as a bare ``Mock``.

    Args:
        mock_analyzer: The mocked ``CodeAnalyzer``; its ``estimate_cost`` is set.
        mock_provider: The mocked provider hanging off ``mock_analyzer.provider``.

    Returns:
        The dict that was installed, for a test that wants to assert on it.
    """
    input_tokens = _as_number(getattr(mock_provider, "total_input_tokens", 0))
    output_tokens = _as_number(getattr(mock_provider, "total_output_tokens", 0))

    pricing = mock_provider.get_pricing.return_value
    input_rate = output_rate = 0.0
    if isinstance(pricing, dict):
        input_rate = _as_number(pricing.get("input_price_per_million", 0))
        output_rate = _as_number(pricing.get("output_price_per_million", 0))

    cost = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": (input_tokens / 1_000_000) * input_rate,
        "output_cost": (output_tokens / 1_000_000) * output_rate,
        "long_context_requests": 0,
    }
    cost["total_cost"] = cost["input_cost"] + cost["output_cost"]
    mock_analyzer.estimate_cost.return_value = cost
    return cost


def _as_number(value: Any) -> float:
    """Coerce a fixture value to a number, treating a bare ``Mock`` as 0."""
    return float(value) if isinstance(value, int | float) else 0.0
