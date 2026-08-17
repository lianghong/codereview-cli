# Pipeline internals

Background for the Architecture and Key-patterns rules in `CLAUDE.md`. Read the section that
covers what you're about to change.

```
FileScanner → FileBatcher → CodeAnalyzer → ProviderFactory → {Bedrock|Azure|NVIDIA|GoogleGenAI|ZAI|DeepSeek|Moonshot|BedrockOpenAI}Provider
            → Aggregation (cli.py) → TerminalRenderer / MarkdownExporter
```

## scanner.py — exclusion runs at two levels, and they are not interchangeable

`_is_excluded` glob-matches each candidate file (correct but walks everything), while
`_get_excluded_dir_names` feeds an `os.walk` prune set that skips whole subtrees.

### Prune names may only come from *unanchored* patterns

**Pruning is by bare directory name, which matches at any depth**, so a name may only be
extracted from an unanchored pattern — every segment before it a wildcard, or none.

`docs/api/*` must not contribute `api`, or an unrelated `app/api/` is skipped too and its files
are never scanned, never counted, and never reported as skipped: the run silently reviews less
code than it claims to. Path-qualified patterns simply don't contribute a prune name; the files
such a pattern actually names are still excluded by `_is_excluded`, just walked rather than
skipped.

Deep descendants a path-qualified pattern doesn't name (`docs/api/sub/x.py` under `docs/api/*`)
are still reviewed — that is the pattern's literal meaning, and getting it wrong the other way
cost coverage in a directory the user never mentioned. To drop a whole subtree, write a recursive
`**`: `docs/api/**` covers arbitrary depth *under that path*, and the unanchored `**/api/**`
covers *any* directory named `api`.

Locked by the four pruning tests in `tests/test_scanner.py`, one of which derives its
expectations from `DEFAULT_EXCLUDE_PATTERNS` so a new default pattern can't quietly change the
prune set.

### `_is_excluded` must test `PurePath.match` *and* `PurePath.full_match`

Neither alone covers the patterns this project ships.

`match` is right-anchored and treats `**` as a *single* segment, so `**/node_modules/**` means
literally "one segment, `node_modules`, one segment": `a/node_modules/x.py` matches, but
`node_modules/x.py` (no leading segment) and `a/b/node_modules/deep/x.py` (too many) do **not** —
and every entry in `DEFAULT_EXCLUDE_PATTERNS` has that shape, so `match` alone left the deep
contents of vendored trees *eligible for review*. The prune set hides that for an ordinary scan
(the directory is never walked) but not when the pattern is path-qualified and so contributes no
prune name, nor for any caller reaching `_is_excluded` directly.

`full_match` recurses `**` correctly but requires the *whole* relative path to match, so it
rejects the right-anchored spellings `match` exists for (`*.py` against `a/b/x.py`) — it cannot
replace `match`.

The union is safe rather than merely convenient: with no `**` in the pattern `full_match` is
*stricter* than `match`, so it can only add matches in the recursive-`**` case, and it does not
widen a path-qualified pattern into another subtree (`docs/api/**` still doesn't match
`app/api/views.py`; a single `*` is one segment under both). The prune set still matters
independently — losing a prune name costs walk time, not coverage, now that `_is_excluded` reads
the recursive patterns correctly.

## batcher.py — token-budget-aware batching

Applies when `context_window` is set; falls back to count-only. Budget =
`context_window − max_output − system_prompt − readme − safety_margin`. Safety margin =
`clamp(context_window // 10, 1000, 20000)`. Token estimate = `bytes // 3 + 50` (tiktoken when
available for files ≤ 2MB; the byte heuristic is the fallback). Greedy packing; oversized files
are skipped with a warning.

Per-file counts are memoized on `(path, size, mtime_ns)` because a single run estimates every
file at least twice (`--dry-run`'s table, then `create_batches`' packing loop) and each estimate
is a full tiktoken encode. **Cache the count, never the content** — batches run concurrently in a
`ThreadPoolExecutor`, so holding every scanned file's text alive for a whole run trades bounded
re-reads for unbounded memory.

`_TOKEN_CACHE_SIZE` is a **run bound, not a "typical repo" bound**: it has to exceed the *file
count*, because at a smaller `maxsize` the first estimate pass evicts its own earliest entries
before the second pass reaches them and every file is re-encoded — the memo silently degrades to
nothing exactly in the large repositories it exists for. It was 4096 (a plausible monorepo file
count); now 100,000, which costs tens of MB against a file list already fully in memory. Locked
by `test_token_cache_bound_exceeds_a_large_repo_file_count`. It's an `lru_cache`, which is
already thread-safe — don't add a lock. Use `clear_token_cache()` in tests that rewrite a file in
place.

## providers/factory.py — dispatch is a table, not an if/elif chain

`_PROVIDER_REGISTRY` maps each provider name to `(config_type, module, class_name)`, and the
config-type guard, the lazy import, the constructor call and the
`Unknown provider: … Supported providers: …` list are all derived from it — so a new provider is
one row and none of those four can disagree with each other.

The module/class are **strings** to keep the import lazy: each provider module imports its
vendor's LangChain client at module scope, so an eager registry would pull all eight client
packages into every run, including `--list-models`, which touches no provider.

Two tests in `tests/test_factory_smoke.py` keep the table honest —
`test_registry_covers_every_provider_models_yaml_configures` (registry set == the set of provider
configs the loader builds) and `test_registry_names_an_importable_provider_class` (each row's
strings resolve to a `ModelProvider` subclass, since a typo in a lazily-imported name otherwise
survives until a user picks that provider's model).

## cli.py — `main()` parses, `run_review()` does the work

`main()` is Click parsing plus the three flags that exit before any review work
(`--list-models`, `--validate`, no-directory help); everything from scanning to the `--fail-on`
gate lives in **`run_review(directory, *, console, ...)`**, which is callable without
`CliRunner.invoke` so tests can assert on ordering and real exception types.

Runs batches concurrently via `ThreadPoolExecutor` (≤4 workers); `--stream` and single-batch runs
stay sequential — but `--stream` is downgraded (with a notice) for a provider that never streams
a token, so it only gives up concurrency where tokens actually appear (see the streaming section
of `docs/providers.md`). Aborts when all batches fail; warns about partial results when some
fail.

`run_review`'s keyword-only params mirror the Click options **including their defaults** —
`main` is a thin pass-through, and
`tests/test_cli.py::test_run_review_defaults_match_the_click_option_defaults` fails if a default
drifts or a new option isn't forwarded. When adding a CLI option, add it to both signatures and
the `run_review(...)` call.

### `--fail-on` is the CI gate, `--severity` is a display filter

Never conflate them. `_evaluate_fail_on` counts `report.issues`, *not* the rendered subset, so
`--severity critical --fail-on high` still exits 2 on a High finding the terminal never showed;
gating on the filtered list would turn a display preference into a silent hole in the gate.
Locked by `tests/test_cli.py::test_severity_filter_does_not_affect_fail_on`.

Exit codes are meaningfully distinct: **1** = the run failed (no results/credentials/API/
unwritable output), **2** = `EXIT_QUALITY_GATE_FAILED`, the review succeeded and found blocking
issues.

The gate is the last statement in `run_review`, *after* export, so a failing build still leaves
its `--output` artifact; `SystemExit` is a `BaseException` so it passes through the
`except Exception` handler while `finally` still cleans up callbacks. That ordering is locked two
ways: `test_fail_on_still_writes_report` (artifact exists after exit 2) and
`test_run_review_applies_the_gate_after_writing_the_report`, which watches the export and the
`SystemExit` happen *in sequence* — the invariant is about order, so at least one test should
assert order rather than an after-the-fact side effect.

### Aggregation must collect every list field the model returns

`issues`, `improvement_suggestions`, `system_design_insights` and `recommendations` all
accumulate across batches. `recommendations` was the one being dropped, and
`_generate_recommendations` substituted severity/category *counts* — so a run that found an SQL
injection at `views.py:42` recommended "🔒 Resolve 1 security issue(s)": no file, no line, no
title, and the same numbers the Metrics section already prints.

`SYSTEM_PROMPT` explicitly asks for recommendations "DERIVED FROM the issues you reported.
Reference issue titles, not new ideas", so the field the prompt works hardest for was the field
thrown away. The model's own text now wins; the counts remain as the **fallback** for runs where
no batch emitted any (partial failures do happen — a generic pointer beats a blank section).
Dedup is on the same lowercased-alphanumerics normalization `_dedupe_design_insights` uses,
because concurrent batches independently reviewing the same shared helper each recommend the same
fix; capped at 5. Locked by the four recommendation tests in `tests/test_cli.py`.

When you add a model-written list field to `CodeReviewReport`, extend the batch loop in the same
commit.

### `files_analyzed` counts what was *sent*, not what was scanned

`sum(len(batch.files) for batch in batches)`, because the batcher drops oversized files after the
scan. Counting `len(files)` claimed coverage the review doesn't have ("Analyzed 120 files" for a
run that reviewed 118), and `--dry-run` priced the dropped bytes too: one skipped multi-MB file
quoted ~1,000,350 input tokens for a run that sends 300. Batch membership is the only authority
on both. Locked by `test_files_analyzed_excludes_files_the_batcher_dropped` and
`test_dry_run_does_not_bill_files_the_batcher_will_drop`.

### Provider error prose is withheld unless `--verbose`

`_aws_error_detail`, cli.py; CWE-209. AWS puts the denying SCP statement, principal/role ARNs,
account and resource ids in `Error.Message`, and `str(ClientError)` splices it in verbatim —
straight into CI logs, which are retained and shared far more widely than the authorization
config they describe. The error **code** is what a user acts on and drives every troubleshooting
hint, so the code always prints and the message is gated.

`providers/bedrock.py`'s `validate_credentials` already did this for its STS/
`ListFoundationModels` branches; the run-time paths (`_render_batch_error` and the top-level
handler) are the loud ones and now match. Don't render a raw `ClientError` or
`error.response[...]["Message"]` unconditionally.

## callbacks.py — one handler serves every concurrent batch

So it is refcounted by `run_id` under a lock with at most one `Status` alive. A single
`self._status` slot that each `on_llm_start` overwrote leaked the previous `Status` with no
reference left to `stop()` it.

Rich's `Console.set_live` *appends to `_live_stack` and returns a bool* rather than raising, so
the overlap raised nothing: it left a permanently-`_started` `Live` on the stack (plus its
refresh thread and a pushed render hook), and the enclosing `Progress`'s `clear_live` then popped
the wrong entry — corrupt terminal state for the rest of the process.

A `set[UUID]` rather than an int makes a duplicate start idempotent and an unmatched end a no-op
instead of an underflow that would stop the spinner while other batches still run. `stop()` is
called outside the lock by the single thread that swapped `_status` to `None`.

## models.py — category normalization

Non-Claude models return varying category names; `@field_validator` maps them (e.g.
`"error handling" → "Code Quality"`). Unknown → `"Code Quality"`.

**A missing category is worse than a mismapped one.** `Correctness` was added 2026-07-25
precisely because it was absent — every word a model naturally reaches for when it finds a bug
(`correctness`, `bug`, `logic`, `race condition`, `crash`, …) coerced to `Code Quality` *and*
bumped `category_coerced`, so real bugs were filed next to naming nits and the drift counter —
whose whole job is detecting prompt drift — was being incremented by correct model behavior.

When adding a category, map its synonyms in the same commit and assert the drift counter stays at
0 (`tests/test_models.py::test_correctness_variations_map_without_drift`). `"error handling"`
deliberately stays on `Code Quality`: as a bare category name it's ambiguous between "this path
crashes" and "use a narrower exception type".

## renderer.py / Markdown export

### Every model-generated string must pass through `balance_code_fences`

Models routinely emit fenced examples inside prose fields, and one unclosed ` ``` ` swallows
*every following section* of the report into a code block — the artifact silently loses its
issues, metrics, and recommendations, which is worse than a visible error because the export
still "succeeds".

The helper counts fence *lines* (first non-whitespace run of ≥3 backticks) and appends the
missing closer; inline triple-backtick spans mid-sentence are deliberately left alone. Applied at
`_summary`, `_format_issue` (description + rationale), `_system_design`, `_recommendations`, and
`_improvement_suggestions`. `suggested_code` is the one exception — it gets a *wider* fence
instead, since it's expected to be code.

When adding a report field the model writes, wrap it here too; locked by the fence tests in
`tests/test_markdown_export.py`. Note this is a **structural** fix, not escaping: deliberately no
HTML/Markdown escaping of model text, since the whole point of the export is readable code and
prose.

### Test on CommonMark fence widths, not fence parity

`tests/test_markdown_export.py::_inside_code_block`. A block opened with N backticks closes only
on a fence line of **at least N** backticks *and* with no info string — so a wider inner fence
(` ````markdown ` inside a ``` wrapper) neither closes the outer block nor opens one of its own,
and a numerically *balanced* document can still trap every following section. A parity assertion
passed under mutation for exactly this reason; the test now runs a real state machine. Assert on
"is this marker inside a code block" and "does the document end inside one", never on backtick
counts.

### Markdown export tolerates raw-dict metrics

`metrics_to_dict` returns `report.metrics` unchanged when it isn't a Pydantic `ReviewMetrics`
(the documented raw-dict fallback), so token values may be stringified or `None`. Every spot that
formats a token metric with `:,` or divides it for cost must `isinstance(..., int)`-guard first —
`_metrics`, the regular-metrics loop, and `_format_summary` all do. Don't add an unguarded
`f"{value:,}"` or cost division on metrics values; locked in by the raw-dict tests in
`tests/test_markdown_export.py`.

### Line counts need the last line even without a trailing newline

Count `"\n"` occurrences *plus one* for a non-empty final line, or every file not ending in a
newline is undercounted by one and the report's `total_lines` drifts low across a repo.

### Pricing display

Zero-priced models (placeholder for unannounced rates) render `Estimated cost: TBD`, not
`$0.0000`. See `_is_pricing_tbd` in `cli.py`.

## `--list-models` rendering

Shows everything regardless of credentials; credentials are only validated when a model is
actually used. It advertises `aliases` only — `deprecated_aliases` collapse to `+N deprecated`
plus a footer, and need `--verbose` to expand.

Rich truncates the Aliases column to `claude-opus-4.…` unless the column sets both
`overflow="fold"` *and* a computed `min_width` (fold alone splits a name across lines at the
80-column default), which would print alias spellings that are invalid as typed; locked by
`tests/test_cli.py::test_list_models_never_truncates_an_alias`. Model dicts from
`list_available_models` are read with `.get("deprecated_aliases", "")` so hand-built test
fixtures without the key still render.
