"""Tiered per-token pricing: the tier belongs to one request, never to a run.

GPT-6 Astra is the first registry entry whose rate depends on how big a single
call is ($11/$55 per M at or below 272K input tokens, $22/$82.50 above). Every
test here exists to pin down the same distinction from a different angle,
because getting it backwards is silent: a run of many cheap batches would be
reported at double its real cost, and CLAUDE.md's rule is that a wrong pricing
number is the worst kind of wrong because the next reader trusts it.
"""

import threading

import pytest
from pydantic import ValidationError

from codereview.cli import _estimate_tiered_cost
from codereview.config.models import ModelConfig, PricingConfig
from codereview.providers.base import ModelProvider
from codereview.providers.mixins import TokenTrackingMixin

FLAT = PricingConfig(input_per_million=11.0, output_per_million=55.0)
TIERED = PricingConfig(
    input_per_million=11.0,
    output_per_million=55.0,
    long_context_threshold_tokens=272_000,
    long_input_per_million=22.0,
    long_output_per_million=82.5,
)


class _Tracker(TokenTrackingMixin):
    """Smallest thing that can carry the mixin: it needs `model_config` only."""

    def __init__(self, pricing: PricingConfig) -> None:
        self.model_config = ModelConfig(
            id="test-model",
            full_id="vendor.test-model",
            name="Test Model",
            pricing=pricing,
        )
        self._init_token_tracking()


# --- PricingConfig.rates_for_request ---------------------------------------


def test_rates_for_request_is_flat_when_no_tier_is_configured():
    assert FLAT.rates_for_request(10) == (11.0, 55.0)
    assert FLAT.rates_for_request(10_000_000) == (11.0, 55.0)
    assert FLAT.has_long_context_tier is False


@pytest.mark.parametrize("input_tokens", [0, 1, 271_999, 272_000])
def test_rates_for_request_stays_cheap_at_and_below_the_threshold(input_tokens):
    """The threshold is inclusive: AWS prices "272K or fewer" at the short rate."""
    assert TIERED.rates_for_request(input_tokens) == (11.0, 55.0)


@pytest.mark.parametrize("input_tokens", [272_001, 500_000, 1_000_000])
def test_rates_for_request_switches_both_rates_above_the_threshold(input_tokens):
    """Crossing the break moves *output* pricing too, not just input."""
    assert TIERED.rates_for_request(input_tokens) == (22.0, 82.5)


def test_a_partial_long_context_tier_is_rejected():
    """Two of three keys is the shape that would price long batches as cheap.

    A missing rate would fall back to the flat pair with no error and no
    warning — the "YAML key that looks like configuration" failure. Rejecting
    it at load time makes the mistake loud.
    """
    with pytest.raises(ValidationError, match="long-context pricing is incomplete"):
        PricingConfig(
            input_per_million=11.0,
            output_per_million=55.0,
            long_context_threshold_tokens=272_000,
            long_input_per_million=22.0,
        )


# --- TokenTrackingMixin: cost accrues per request --------------------------


def test_a_multi_batch_run_is_not_charged_the_long_tier_it_never_reached():
    """Five 100K requests total 500K tokens and still bill entirely at $11/$55.

    This is the whole reason tier selection lives in ``_track_tokens`` rather
    than ``estimate_cost``: by the time only the 500K total remains, the fact
    that no single call was over 272K is unrecoverable.
    """
    tracker = _Tracker(TIERED)
    for _ in range(5):
        tracker._track_tokens(100_000, 10_000)

    cost = tracker.estimate_cost()

    assert cost["input_tokens"] == 500_000
    assert cost["long_context_requests"] == 0
    assert cost["input_cost"] == pytest.approx(0.5 * 11.0)
    assert cost["output_cost"] == pytest.approx(0.05 * 55.0)


def test_only_the_requests_over_the_threshold_pay_the_long_rate():
    """A mixed run blends: each request keeps the rate its own size earned."""
    tracker = _Tracker(TIERED)
    tracker._track_tokens(100_000, 10_000)  # short tier
    tracker._track_tokens(300_000, 20_000)  # long tier

    cost = tracker.estimate_cost()

    assert cost["long_context_requests"] == 1
    assert tracker.long_context_requests == 1
    assert cost["input_cost"] == pytest.approx(0.1 * 11.0 + 0.3 * 22.0)
    assert cost["output_cost"] == pytest.approx(0.01 * 55.0 + 0.02 * 82.5)


def test_per_request_accrual_equals_the_old_flat_multiplication():
    """The twenty untiered entries must be unaffected, to the cent.

    ``sum(tokens_i) * rate == sum(tokens_i * rate)``, so switching from one
    multiplication over the totals to an accumulator cannot move an untiered
    model's reported cost.
    """
    tracker = _Tracker(FLAT)
    sizes = [(1_000, 200), (350_000, 40_000), (17, 3)]
    for input_tokens, output_tokens in sizes:
        tracker._track_tokens(input_tokens, output_tokens)

    cost = tracker.estimate_cost()
    expected_in = sum(i for i, _ in sizes) / 1_000_000 * 11.0
    expected_out = sum(o for _, o in sizes) / 1_000_000 * 55.0

    assert cost["input_cost"] == pytest.approx(expected_in)
    assert cost["output_cost"] == pytest.approx(expected_out)
    assert cost["total_cost"] == pytest.approx(expected_in + expected_out)
    assert cost["long_context_requests"] == 0


def test_reset_state_zeroes_the_accrued_cost_and_the_tier_counter():
    """A stale accrual would carry one run's cost into the next."""
    tracker = _Tracker(TIERED)
    tracker._track_tokens(400_000, 30_000)
    assert tracker.estimate_cost()["total_cost"] > 0

    tracker.reset_state()
    cost = tracker.estimate_cost()

    assert cost == {
        "input_tokens": 0,
        "output_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "long_context_requests": 0,
    }


def test_concurrent_batches_accrue_cost_without_losing_an_increment():
    """Batches run in a ThreadPoolExecutor; the accrual shares their lock."""
    tracker = _Tracker(TIERED)
    threads = [
        threading.Thread(target=tracker._track_tokens, args=(300_000, 1_000))
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    cost = tracker.estimate_cost()
    assert cost["long_context_requests"] == 8
    assert cost["input_cost"] == pytest.approx(8 * 0.3 * 22.0)


# --- get_pricing() reports the tier additively -----------------------------


def test_get_pricing_omits_the_tier_keys_entirely_for_a_flat_model():
    """Absent, not null: consumers ``.get()`` and a None would poison a float."""
    rates = ModelProvider.get_pricing(_Tracker(FLAT))

    assert rates == {"input_price_per_million": 11.0, "output_price_per_million": 55.0}


def test_get_pricing_exposes_all_three_tier_keys_for_a_tiered_model():
    rates = ModelProvider.get_pricing(_Tracker(TIERED))

    assert rates["input_price_per_million"] == 11.0
    assert rates["output_price_per_million"] == 55.0
    assert rates["long_context_threshold_tokens"] == 272_000
    assert rates["long_input_price_per_million"] == 22.0
    assert rates["long_output_price_per_million"] == 82.5


# --- The --dry-run estimator ----------------------------------------------


class _FakeBatch:
    """Stands in for ``FileBatch``; ``_estimate_tiered_cost`` reads `.files`."""

    def __init__(self, files):
        self.files = files


def _pricing_dict(*, tiered: bool) -> dict[str, float]:
    rates = {"input_price_per_million": 11.0, "output_price_per_million": 55.0}
    if tiered:
        rates["long_context_threshold_tokens"] = 272_000
        rates["long_input_price_per_million"] = 22.0
        rates["long_output_price_per_million"] = 82.5
    return rates


def test_dry_run_prices_each_batch_as_its_own_request(monkeypatch):
    """Four 100K batches must not be summed into one 400K long-context call.

    The old flat estimator summed every batch's input before applying one rate,
    which is correct for a flat model and a 2x overstatement here.
    """
    monkeypatch.setattr(
        "codereview.cli.FileBatcher.estimate_file_tokens",
        staticmethod(lambda path: 100_000),
    )
    batches = [_FakeBatch([f"f{i}.py"]) for i in range(4)]

    input_cost, output_cost, long_batches = _estimate_tiered_cost(
        batches, 0, _pricing_dict(tiered=True)
    )

    assert long_batches == 0
    assert input_cost == pytest.approx(4 * 0.1 * 11.0)
    assert output_cost == pytest.approx(4 * 0.02 * 55.0)


def test_dry_run_charges_the_long_tier_for_the_batches_that_earn_it(monkeypatch):
    """Per-batch overhead counts toward the threshold — it is billed input."""
    sizes = {"small.py": 50_000, "big.py": 271_000}
    monkeypatch.setattr(
        "codereview.cli.FileBatcher.estimate_file_tokens",
        staticmethod(lambda path: sizes[path]),
    )
    # 271_000 + 2_000 overhead crosses 272_000; 50_000 + 2_000 does not.
    batches = [_FakeBatch(["small.py"]), _FakeBatch(["big.py"])]

    input_cost, output_cost, long_batches = _estimate_tiered_cost(
        batches, 2_000, _pricing_dict(tiered=True)
    )

    assert long_batches == 1
    assert input_cost == pytest.approx(0.052 * 11.0 + 0.273 * 22.0)
    assert output_cost == pytest.approx(
        int(52_000 * 0.2) / 1_000_000 * 55.0 + int(273_000 * 0.2) / 1_000_000 * 82.5
    )


def test_dry_run_estimate_is_unchanged_for_an_untiered_model(monkeypatch):
    """No tier keys in ``get_pricing()`` means the flat arithmetic, exactly."""
    monkeypatch.setattr(
        "codereview.cli.FileBatcher.estimate_file_tokens",
        staticmethod(lambda path: 400_000),
    )
    batches = [_FakeBatch(["a.py"]), _FakeBatch(["b.py"])]

    input_cost, output_cost, long_batches = _estimate_tiered_cost(
        batches, 0, _pricing_dict(tiered=False)
    )

    assert long_batches == 0
    assert input_cost == pytest.approx(0.8 * 11.0)
    assert output_cost == pytest.approx(0.16 * 55.0)
