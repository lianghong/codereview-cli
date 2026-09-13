"""Tiered per-token pricing: the tier belongs to one request, never to a run.

GPT-6 Astra is the first registry entry whose rate depends on how big a single
call is ($11/$55 per M at or below 272K input tokens, $22/$82.50 above). Every
test here exists to pin down the same distinction from a different angle,
because getting it backwards is silent: a run of many cheap batches would be
reported at double its real cost, and CLAUDE.md's rule is that a wrong pricing
number is the worst kind of wrong because the next reader trusts it.
"""

import re
import threading
from pathlib import Path

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


# ---------------------------------------------------------------------------
# The consumers: every cost figure a user sees must be the accrual
# ---------------------------------------------------------------------------
#
# `--dry-run` was made tier-aware first and the *real* run's summary was left
# recomputing `provider.total_input_tokens * flat_rate`, with the Markdown
# export doing the same independently — so `estimate_cost()`, the only
# tier-aware figure, had no caller at all. Unclamping Astra's window made
# batches over the threshold reachable and turned that into a live
# under-report. The tests below deliberately hand the CLI an accrual that
# DISAGREES with the flat arithmetic, so anyone who recomputes fails here.


def _disagreeing_cost() -> dict[str, float | int]:
    """An accrual no flat multiplication of the totals can produce.

    100K in / 20K out at the short rates would be $1.10 + $1.10 = $2.20. This
    says $2.20 + $1.65 = one request that crossed 272K, billed at $22/$82.50.
    """
    return {
        "input_tokens": 100_000,
        "output_tokens": 20_000,
        "input_cost": 2.2,
        "output_cost": 1.65,
        "total_cost": 3.85,
        "long_context_requests": 1,
    }


def _run_review_output(tmp_path) -> str:
    """Run ``run_review`` over one file with a tiered accrual, return output."""
    from io import StringIO
    from unittest.mock import Mock, patch

    from rich.console import Console

    from codereview.cli import run_review
    from codereview.models import CodeReviewReport, ReviewMetrics

    (tmp_path / "test.py").write_text("def hello():\n    return 'world'\n")

    cost = _disagreeing_cost()
    provider = Mock()
    provider.total_input_tokens = cost["input_tokens"]
    provider.total_output_tokens = cost["output_tokens"]
    # The flat rates: only the SHORT-context ones are exposed here, which is
    # exactly why a consumer must not multiply by them.
    provider.get_pricing.return_value = {
        "input_price_per_million": 11.0,
        "output_price_per_million": 55.0,
    }

    analyzer = Mock()
    analyzer.provider = provider
    analyzer.estimate_cost.return_value = cost
    analyzer.skipped_files = []
    analyzer.analyze_batch.return_value = CodeReviewReport(
        summary="Test",
        metrics=ReviewMetrics(files_analyzed=1),
        issues=[],
        system_design_insights="None",
        recommendations=[],
        improvement_suggestions=[],
    )

    buffer = StringIO()
    with patch("codereview.cli.CodeAnalyzer", return_value=analyzer):
        run_review(tmp_path, console=Console(file=buffer, width=200), no_readme=True)
    return buffer.getvalue()


def test_run_review_prints_the_accrued_cost_not_tokens_times_the_flat_rate(tmp_path):
    output = _run_review_output(tmp_path)
    assert "$3.8500" in output, (
        "the summary did not print the provider's accrual. If it shows "
        "$2.2000 it is recomputing totals * the short-context rate — the "
        "under-report this test exists to prevent."
    )
    assert "$2.2000" not in output


def test_run_review_names_the_requests_billed_at_the_long_tier(tmp_path):
    """A doubled rate the user cannot see is indistinguishable from a bug."""
    output = _run_review_output(tmp_path)
    assert "1 request(s) billed" in output
    assert "long-context" in output


def _blank_report(metrics):
    from codereview.models import CodeReviewReport

    return CodeReviewReport(
        summary="Test",
        metrics=metrics,
        issues=[],
        system_design_insights="None",
        recommendations=[],
        improvement_suggestions=[],
    )


def _export_report(report, tmp_path) -> str:
    from codereview.renderer import MarkdownExporter

    out = tmp_path / "report.md"
    MarkdownExporter().export(report, out)
    return out.read_text(encoding="utf-8")


def _export(metrics_dict, tmp_path) -> str:
    from codereview.models import ReviewMetrics

    return _export_report(_blank_report(ReviewMetrics(**metrics_dict)), tmp_path)


_EXPORT_METRICS = {
    "files_analyzed": 1,
    "input_tokens": 100_000,
    "output_tokens": 20_000,
    "total_tokens": 120_000,
    "input_price_per_million": 11.0,
    "output_price_per_million": 55.0,
}


def test_the_export_reports_the_accrual_and_drops_the_per_m_annotation(tmp_path):
    """`(\\$11.00/M tokens)` next to a long-tier figure would be a lie."""
    markdown = _export(
        {
            **_EXPORT_METRICS,
            "input_cost": 2.2,
            "output_cost": 1.65,
            "long_context_requests": 1,
        },
        tmp_path,
    )
    assert "$3.8500 USD" in markdown
    assert "1 request(s) billed at long-context rates" in markdown
    assert "/M tokens" not in markdown


def test_the_export_keeps_the_per_m_annotation_for_an_untiered_run(tmp_path):
    """Zero long requests is the common case and must read exactly as before."""
    markdown = _export(
        {
            **_EXPORT_METRICS,
            "input_cost": 1.1,
            "output_cost": 1.1,
            "long_context_requests": 0,
        },
        tmp_path,
    )
    assert "$2.2000 USD" in markdown
    assert "($11.00/M tokens)" in markdown


def test_the_export_falls_back_to_flat_arithmetic_without_an_accrual(tmp_path):
    """A hand-built or pre-accrual metrics dict still gets a cost.

    No provider ran, so there is nothing to report and the flat product is
    both all that is available and exact for an untiered entry.
    """
    markdown = _export(_EXPORT_METRICS, tmp_path)
    assert "$2.2000 USD" in markdown
    assert "($11.00/M tokens)" in markdown


def test_the_export_tolerates_a_raw_dict_whose_accrual_is_a_string(tmp_path):
    """`metrics_to_dict` passes a non-ReviewMetrics through untouched.

    So the accrual keys can hold anything, and the renderer has to fall back
    rather than ``float()`` a string.
    """
    from codereview.models import ReviewMetrics

    report = _blank_report(ReviewMetrics(files_analyzed=1))
    report.metrics = {**_EXPORT_METRICS, "input_cost": "n/a", "output_cost": None}
    markdown = _export_report(report, tmp_path)
    assert "$2.2000 USD" in markdown


# ---------------------------------------------------------------------------
# Source guard: no new consumer may recompute the cost
# ---------------------------------------------------------------------------

# Every place in `codereview/` allowed to turn tokens into money by
# multiplying a rate, with the reason. Anything else must read the accrual
# from `estimate_cost()` / the metrics, because the rate that applied is a
# property of one request and is not recoverable from a total.
_ALLOWED_COST_ARITHMETIC = {
    "providers/mixins.py": (
        "_track_tokens is where the accrual is PRODUCED — the only place one "
        "request's input size is still visible, so the only place the tier "
        "can be chosen."
    ),
    "cli.py": (
        "_estimate_tiered_cost does --dry-run, where there are no requests "
        "yet, so it prices each planned batch itself."
    ),
    "renderer.py": (
        "the documented flat fallback for a metrics dict carrying no accrual "
        "(hand-built or legacy); exact for an untiered entry."
    ),
}

_COST_ARITHMETIC = re.compile(
    r"/\s*1_000_000\s*\)?\s*\*|\*\s*\(?[\w.\[\]\"']+\s*/\s*1_000_000"
)


def test_no_new_consumer_recomputes_cost_from_a_rate():
    """Dividing by 1_000_000 and multiplying is how the bug was written twice.

    Both the CLI summary and the Markdown export independently recomputed
    `tokens / 1_000_000 * rate` off the run *totals* while the tier-aware
    accrual had no caller at all. This guard makes the third occurrence a
    failing test rather than a wrong number in a report.
    """
    package = Path(__file__).resolve().parent.parent / "codereview"
    offenders = sorted(
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if _COST_ARITHMETIC.search(path.read_text(encoding="utf-8"))
        and str(path.relative_to(package)) not in _ALLOWED_COST_ARITHMETIC
    )
    assert not offenders, (
        "these files compute money from a per-million rate: "
        f"{offenders}. The rate that applied depends on ONE request's input "
        "size, so a total cannot be repriced — read `input_cost`/"
        "`output_cost` off estimate_cost() or the metrics instead. If the "
        "site genuinely has no accrual to read (a dry run, a provider-side "
        "producer), add it to _ALLOWED_COST_ARITHMETIC with the reason."
    )


def test_the_cost_arithmetic_allowlist_has_no_stale_entries():
    """An allowlisted file that stopped doing the arithmetic must be dropped."""
    package = Path(__file__).resolve().parent.parent / "codereview"
    stale = sorted(
        name
        for name in _ALLOWED_COST_ARITHMETIC
        if not _COST_ARITHMETIC.search((package / name).read_text(encoding="utf-8"))
    )
    assert not stale, (
        f"_ALLOWED_COST_ARITHMETIC names files that no longer price tokens: "
        f"{stale}. Remove the entry so the allowlist keeps meaning something."
    )
