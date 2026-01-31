# README Context for Code Reviews

**Date:** 2026-02-01
**Status:** Approved

## Overview

Add project README.md content as context for AI code reviews. The README is automatically discovered, confirmed by the user once per session, and included in each batch sent to the LLM.

## User Flow

### README Found

```
$ codereview ./src

🔍 Code Review Tool

📄 Found README: /home/user/project/README.md (2.3 KB)
   Use this file for project context? [Y/n/path]:

   Y (or Enter) → Use this README
   n            → Skip README context
   path         → Use specified file instead
```

### README Not Found

```
📄 No README.md found in ./src or parent directories
   Specify a file for project context? [path/N]:
```

## Search Algorithm

```
Target: ./project/src/components
Search order:
  1. ./project/src/components/README.md
  2. ./project/src/README.md
  3. ./project/README.md
  4. (stop at filesystem root or git root)
```

- Stops at first README.md found
- Stops at git repository root to avoid searching outside project
- Only searches for README.md (not other variants)

## Integration Points

### New Module: `codereview/readme_finder.py`

- `find_readme(target_dir: Path) -> Path | None` — Search logic
- `prompt_readme_confirmation(readme_path: Path | None) -> Path | None` — User interaction

### Modified: `codereview/cli.py`

- Call readme finder after directory validation
- Store confirmed README path
- Pass to provider factory

### Modified: `codereview/providers/base.py`

- Add `project_context: str | None` parameter to `_prepare_batch_context()`
- Prepend README content before file contents when present

## Batch Context Format

```
== PROJECT CONTEXT ==
The following is the project README for background context:

--- README.md ---
[README content here]
--- END README ---

== CODE REVIEW ==
Analyzing Batch 1/3
Files in this batch: 5

================================================================================

File: src/main.py
--------------------------------------------------------------------------------
   1 | import sys
   2 | ...
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--readme <path>` | Specify README file directly (skips auto-discovery and prompt) |
| `--no-readme` | Skip README context entirely (no prompt) |

These enable non-interactive usage for CI/CD pipelines.

## Edge Cases & Limits

### Large README Handling

- Warn if README > 50KB (~12,500 tokens)
- Truncate with notice if > 100KB
- Show size in confirmation prompt

### Error Handling

- File read errors → warn and continue without README
- Binary/non-text files → reject with message
- Permission denied → warn and continue

### Session Memory

- README preference stored in memory only
- Applies to current CLI invocation
- Multiple directories reuse confirmed README

## Implementation Tasks

1. Create `codereview/readme_finder.py` with search and prompt logic
2. Add `--readme` and `--no-readme` CLI options
3. Modify `_prepare_batch_context()` in base provider to accept project context
4. Update CLI to integrate readme finder into main flow
5. Add tests for readme finder module
6. Update CLAUDE.md documentation
