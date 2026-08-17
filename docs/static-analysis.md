# Static Analysis Tools Guide

This document provides comprehensive information about all static analysis tools supported by the codereview CLI, including installation guides, configuration options, and usage examples.

## Overview

The codereview CLI integrates with static analysis tools across multiple languages to provide comprehensive code quality and security checks. Tools run in parallel using `ThreadPoolExecutor` for optimal performance.

### Supported Languages

| Language | Tools |
|----------|-------|
| Python | ruff, mypy, black, isort, vulture, bandit |
| Go | golangci-lint, go vet, gofmt, gosec |
| Shell/Bash | shellcheck, bashate |
| C++ | clang-tidy, cppcheck, clang-format |
| Java | checkstyle |
| JavaScript/TypeScript | eslint, prettier, tsc, npm audit |

### Running Static Analysis

```bash
# Run AI review with static analysis
uv run codereview /path/to/code --static-analysis

# Combined with other options
uv run codereview ./src -m sonnet --static-analysis --output report.md
```

### Repo-Supplied Configs That Execute Code

Three of these tools load code the *reviewed repository* provides:

| Tool | Config | What it executes |
|---|---|---|
| **Mypy** | `mypy.ini`, `.mypy.ini`, `setup.cfg`, `pyproject.toml` with a `plugins =` entry in the mypy section | imports the named Python module |
| **ESLint** | `eslint.config.{js,mjs,cjs,ts}`, `.eslintrc.{js,cjs,mjs}` | the config file *is* JavaScript |
| **ESLint / Prettier** | `.eslintrc.json`/`.eslintrc.yaml`, `.prettierrc*`, `package.json` with a `plugins` key | loads the named plugin module |

Reviewing an untrusted repository would otherwise run that code with your
privileges. By default the affected tool is **skipped** with a visible reason;
`--trust-repo-config` opts back in:

```bash
uv run codereview ./src --static-analysis --trust-repo-config
```

Detection is on content, not filename: an ordinary `pyproject.toml` with a
`[tool.mypy]` section but no `plugins` entry still runs mypy. Configs that can't
be read, or that exceed 512 KB, are treated as risky.

---

## Python Tools

### Ruff

**Purpose:** Fast Python linter written in Rust. Replaces Flake8, isort, pydocstyle, and more.

**Installation:**
```bash
pip install ruff
# or with uv
uv pip install ruff
```

**Configuration:** Create `ruff.toml` or `pyproject.toml`:
```toml
[tool.ruff]
line-length = 88
select = ["E", "F", "W", "I", "N", "B", "A", "C4", "SIM"]
ignore = ["E501"]  # line too long

[tool.ruff.per-file-ignores]
"tests/*" = ["S101"]  # allow assert in tests
```

**Manual Usage:**
```bash
ruff check .                    # Check for issues
ruff check --fix .              # Auto-fix issues
ruff check --select E,F .       # Select specific rules
```

---

### Mypy

**Purpose:** Static type checker for Python. Enforces type annotations.

**Installation:**
```bash
pip install mypy types-PyYAML types-requests  # Include type stubs
# or with uv
uv pip install mypy types-PyYAML
```

**Configuration:** Create `mypy.ini` or `pyproject.toml`:
```toml
[tool.mypy]
python_version = "3.14"
strict = true
warn_return_any = true
warn_unused_ignores = true
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "tests.*"
ignore_errors = true
```

**Manual Usage:**
```bash
mypy src/                       # Check directory
mypy --strict src/              # Strict mode
mypy --ignore-missing-imports . # Ignore missing stubs
```

---

### Black

**Purpose:** Uncompromising Python code formatter. Enforces consistent style.

**Installation:**
```bash
pip install black
# or with uv
uv pip install black
```

**Configuration:** In `pyproject.toml`:
```toml
[tool.black]
line-length = 88
target-version = ['py314']
include = '\.pyi?$'
exclude = '''
/(
    \.git
  | \.venv
  | build
  | dist
)/
'''
```

**Manual Usage:**
```bash
black .                         # Format all files
black --check .                 # Check without modifying
black --diff .                  # Show what would change
```

---

### isort

**Purpose:** Sorts Python imports alphabetically and by section.

**Installation:**
```bash
pip install isort
# or with uv
uv pip install isort
```

**Configuration:** In `pyproject.toml`:
```toml
[tool.isort]
profile = "black"
line_length = 88
known_first_party = ["codereview"]
skip = [".venv", "build"]
```

**Manual Usage:**
```bash
isort .                         # Sort imports
isort --check-only .            # Check without modifying
isort --diff .                  # Show changes
```

---

### Vulture

**Purpose:** Finds unused code (dead code) in Python projects.

**Installation:**
```bash
pip install vulture
# or with uv
uv pip install vulture
```

**Configuration:** Create `vulture_whitelist.py` for false positives:
```python
# vulture_whitelist.py
_.unused_variable  # Mark as used
SomeClass.unused_method  # Whitelist method
```

**Manual Usage:**
```bash
vulture src/                    # Find dead code
vulture --min-confidence 80 .   # Set confidence threshold
vulture src/ whitelist.py       # Use whitelist
```

---

### Bandit (Security Scanner)

**Purpose:** Security-focused static analyzer for Python. Detects common security issues.

**Installation:**
```bash
pip install bandit
# or with uv
uv pip install bandit
```

**What It Detects:**
- Hardcoded passwords and secrets
- SQL injection vulnerabilities
- Command injection via subprocess
- Weak cryptographic functions (MD5, SHA1)
- Use of assert in production code
- Insecure temp file creation
- Binding to all interfaces (0.0.0.0)

**Configuration:** Create `.bandit` or `pyproject.toml`:
```toml
[tool.bandit]
exclude_dirs = ["tests", "venv"]
skips = ["B101"]  # Skip assert warnings

[tool.bandit.assert_used]
skips = ["*_test.py", "test_*.py"]
```

**Manual Usage:**
```bash
bandit -r src/                  # Recursive scan
bandit -r -f txt src/           # Text output format
bandit -r -ll src/              # Only high severity
bandit -r -c .bandit src/       # Use config file
bandit -r --severity-level medium src/  # Filter by severity
```

**Severity Levels:**
- **HIGH:** Critical security issues (hardcoded passwords, SQL injection)
- **MEDIUM:** Potential vulnerabilities (weak crypto, insecure defaults)
- **LOW:** Code quality issues with security implications

---

## Go Tools

### golangci-lint

**Purpose:** Meta-linter aggregating 50+ linters for Go. The recommended Go linter.

**Installation:**
```bash
# Binary installation (recommended)
curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/HEAD/install.sh | sh -s -- -b $(go env GOPATH)/bin

# Or via go install
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
```

**Configuration:** Create `.golangci.yml`:
```yaml
linters:
  enable:
    - errcheck
    - gosimple
    - govet
    - ineffassign
    - staticcheck
    - unused
    - gosec

linters-settings:
  errcheck:
    check-blank: true
  govet:
    enable-all: true

issues:
  exclude-dirs:
    - vendor
    - testdata
```

**Manual Usage:**
```bash
golangci-lint run                # Run all enabled linters
golangci-lint run --fix          # Auto-fix where possible
golangci-lint run --enable-all   # Enable all linters
golangci-lint linters            # List available linters
```

---

### go vet

**Purpose:** Built-in Go static analyzer. Reports suspicious constructs.

**Installation:** Included with Go.

**What It Detects:**
- Printf format string mismatches
- Unreachable code
- Suspicious assignments
- Invalid struct tags
- Shadowed variables

**Manual Usage:**
```bash
go vet ./...                    # Check all packages
go vet -json ./...              # JSON output
go vet -vettool=staticcheck .   # Use external analyzer
```

---

### gofmt

**Purpose:** Official Go code formatter. Enforces Go formatting standards.

**Installation:** Included with Go.

**Manual Usage:**
```bash
gofmt -l .                      # List unformatted files
gofmt -d .                      # Show diffs
gofmt -w .                      # Format in place
gofmt -s -w .                   # Simplify and format
```

---

### gosec (Security Scanner)

**Purpose:** Go security checker. Inspects source code for security vulnerabilities.

**Installation:**
```bash
go install github.com/securego/gosec/v2/cmd/gosec@latest
```

**What It Detects:**
- G101: Hardcoded credentials
- G102: Bind to all interfaces
- G103: Unsafe block usage
- G104: Audit errors not checked
- G107: URL provided to HTTP request
- G201-204: SQL injection vulnerabilities
- G301-307: File permission issues
- G401-404: Weak cryptographic primitives
- G501-505: Blocklisted imports

**Configuration:** Create `.gosec.yaml`:
```yaml
global:
  audit: enabled
  nosec: enabled

rules:
  exclude:
    - G104  # Exclude unhandled errors
```

**Manual Usage:**
```bash
gosec ./...                     # Scan all packages
gosec -fmt=json ./...           # JSON output
gosec -severity medium ./...    # Filter by severity
gosec -exclude=G104 ./...       # Exclude specific rules
gosec -conf .gosec.yaml ./...   # Use config file
```

**Severity Levels:**
- **HIGH:** Critical vulnerabilities (SQL injection, hardcoded creds)
- **MEDIUM:** Potential security issues (weak crypto, file permissions)
- **LOW:** Informational findings

---

## Shell Script Tools

### ShellCheck

**Purpose:** Static analysis tool for shell scripts. Catches common bugs and pitfalls.

**Installation:**
```bash
# Ubuntu/Debian
sudo apt install shellcheck

# macOS
brew install shellcheck

# Fedora
sudo dnf install ShellCheck
```

**What It Detects:**
- SC1000-1999: Syntax issues
- SC2000-2999: Warning-level issues (quoting, word splitting)
- SC3000-3999: Bash-specific features in sh scripts

**Configuration:** Create `.shellcheckrc`:
```ini
# Disable specific checks
disable=SC2034,SC2086

# Set default shell dialect
shell=bash

# Enable optional checks
enable=require-variable-braces
```

**Manual Usage:**
```bash
shellcheck script.sh            # Check single file
shellcheck -x script.sh         # Follow source directives
shellcheck -f json *.sh         # JSON output
shellcheck -S warning *.sh      # Only warnings and above
shellcheck -e SC2086 script.sh  # Exclude specific check
```

**Common Checks:**
- SC2086: Double quote to prevent globbing/word splitting
- SC2046: Quote command substitutions
- SC2034: Variable appears unused
- SC2164: Use cd ... || exit

---

### bashate

**Purpose:** PEP8-style checker for bash scripts. Enforces consistent style.

**Installation:**
```bash
pip install bashate
# or with uv
uv pip install bashate
```

**What It Checks:**
- E001: Trailing whitespace
- E002: Tab indentation
- E003: Indent not multiple of 4
- E004: File did not end with newline
- E005: File does not begin with #!
- E006: Line too long
- E010: do not on same line as for
- E011: then not on same line as if
- E020: Function declaration not in format `name()`
- E040: Syntax error
- E041: Arithmetic expansion using deprecated syntax
- E042: Local declaration hides errors
- E043: Arithmetic compound has inconsistent return semantics

**Configuration:** Use command-line flags or create a tox.ini:
```ini
[bashate]
ignore = E006
max_line_length = 120
```

**Manual Usage:**
```bash
bashate script.sh               # Check single file
bashate *.sh                    # Check all shell scripts
bashate -i E006 script.sh       # Ignore specific error
bashate --max-line-length 120 . # Set line length
bashate -v script.sh            # Verbose output
```

---

## C++ Tools

### clang-tidy

**Purpose:** Clang-based C++ linter and static analyzer. Part of LLVM.

**Installation:**
```bash
# Ubuntu/Debian
sudo apt install clang-tidy

# macOS
brew install llvm
# Add to PATH: export PATH="/usr/local/opt/llvm/bin:$PATH"
```

**Configuration:** Create `.clang-tidy`:
```yaml
Checks: >
  -*,
  bugprone-*,
  cert-*,
  cppcoreguidelines-*,
  modernize-*,
  performance-*,
  readability-*
WarningsAsErrors: ''
HeaderFilterRegex: ''
FormatStyle: file
```

**Manual Usage:**
```bash
clang-tidy src/*.cpp            # Check files
clang-tidy -p build/ src/*.cpp  # Use compile_commands.json
clang-tidy --fix src/*.cpp      # Auto-fix issues
clang-tidy --checks='-*,modernize-*' src/*.cpp  # Specific checks
```

---

### cppcheck

**Purpose:** Static analysis tool for C/C++. Detects bugs and undefined behavior.

**Installation:**
```bash
# Ubuntu/Debian
sudo apt install cppcheck

# macOS
brew install cppcheck
```

**What It Detects:**
- Memory leaks and buffer overflows
- Null pointer dereferences
- Division by zero
- Resource leaks
- Uninitialized variables
- Dead code

**Configuration:** Create `cppcheck.cfg` or use command-line:
```xml
<?xml version="1.0"?>
<cppcheck>
  <suppress>
    <id>missingIncludeSystem</id>
  </suppress>
</cppcheck>
```

**Manual Usage:**
```bash
cppcheck src/                   # Check directory
cppcheck --enable=all src/      # Enable all checks
cppcheck --xml src/             # XML output
cppcheck --suppress=missingInclude src/  # Suppress warning
cppcheck --force src/           # Check even with errors
```

---

### clang-format

**Purpose:** Code formatter for C/C++/Java/JavaScript/Objective-C.

**Installation:**
```bash
# Ubuntu/Debian
sudo apt install clang-format

# macOS
brew install clang-format
```

**Configuration:** Create `.clang-format`:
```yaml
BasedOnStyle: Google
IndentWidth: 4
ColumnLimit: 100
AllowShortFunctionsOnASingleLine: None
BreakBeforeBraces: Attach
```

**Manual Usage:**
```bash
clang-format -i src/*.cpp       # Format in place
clang-format --dry-run src/*.cpp  # Check without modifying
clang-format --style=Google src/*.cpp  # Use style preset
clang-format -style=file src/*.cpp  # Use .clang-format file
```

---

## Java Tools

### Checkstyle

**Purpose:** Development tool to ensure Java code adheres to coding standards.

**Installation:**
```bash
# Download JAR
curl -LO https://github.com/checkstyle/checkstyle/releases/download/checkstyle-10.12.5/checkstyle-10.12.5-all.jar

# Or via package manager (Ubuntu)
sudo apt install checkstyle
```

**Configuration:** Use built-in configs or create `checkstyle.xml`:
```xml
<?xml version="1.0"?>
<!DOCTYPE module PUBLIC
  "-//Checkstyle//DTD Checkstyle Configuration 1.3//EN"
  "https://checkstyle.org/dtds/configuration_1_3.dtd">
<module name="Checker">
  <module name="TreeWalker">
    <module name="AvoidStarImport"/>
    <module name="ConstantName"/>
    <module name="EmptyBlock"/>
    <module name="EqualsHashCode"/>
  </module>
</module>
```

**Manual Usage:**
```bash
# Using JAR
java -jar checkstyle.jar -c /google_checks.xml src/

# Using system installation
checkstyle -c /google_checks.xml src/*.java

# Custom config
checkstyle -c checkstyle.xml src/
```

**Built-in Configs:**
- `/google_checks.xml` - Google Java Style Guide
- `/sun_checks.xml` - Sun Code Conventions

---

## JavaScript/TypeScript Tools

### ESLint

**Purpose:** Pluggable linting utility for JavaScript and TypeScript.

**Installation:**
```bash
npm install -g eslint
# Or locally
npm install --save-dev eslint
```

**Configuration:** Create `eslint.config.js` (flat config) or `.eslintrc.json`:
```javascript
// eslint.config.js (ESLint 9+)
export default [
  {
    rules: {
      "no-unused-vars": "error",
      "no-console": "warn",
      "semi": ["error", "always"]
    }
  }
];
```

```json
// .eslintrc.json (legacy)
{
  "env": {
    "browser": true,
    "node": true,
    "es2024": true
  },
  "extends": ["eslint:recommended"],
  "rules": {
    "no-unused-vars": "error",
    "semi": ["error", "always"]
  }
}
```

**Manual Usage:**
```bash
eslint src/                     # Check directory
eslint --fix src/               # Auto-fix issues
eslint --ext .js,.ts src/       # Specify extensions
eslint --format json src/       # JSON output
```

---

### Prettier

**Purpose:** Opinionated code formatter for JS/TS/CSS/HTML/JSON/Markdown.

**Installation:**
```bash
npm install -g prettier
# Or locally
npm install --save-dev prettier
```

**Configuration:** Create `.prettierrc`:
```json
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "es5",
  "printWidth": 100
}
```

**Manual Usage:**
```bash
prettier --check .              # Check formatting
prettier --write .              # Format in place
prettier --single-quote src/    # Override config
```

---

### TypeScript Compiler (tsc)

**Purpose:** TypeScript type checker and compiler.

**Installation:**
```bash
npm install -g typescript
# Or locally
npm install --save-dev typescript
```

**Configuration:** Create `tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "strict": true,
    "noEmit": true,
    "skipLibCheck": true,
    "esModuleInterop": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules"]
}
```

**Manual Usage:**
```bash
tsc --noEmit                    # Type check only
tsc --project tsconfig.json     # Use specific config
tsc --strict src/*.ts           # Strict mode
```

---

### npm audit (Security Scanner)

**Purpose:** Scans dependencies for known security vulnerabilities.

**Installation:** Included with npm (comes with Node.js).

**What It Detects:**
- Known vulnerabilities in dependencies
- Outdated packages with security patches
- Transitive dependency vulnerabilities

**Manual Usage:**
```bash
npm audit                       # Show vulnerabilities
npm audit --json                # JSON output
npm audit fix                   # Auto-fix where possible
npm audit fix --force           # Force updates (may break)
npm audit --production          # Only production deps
```

**Severity Levels:**
- **Critical:** Immediate action required
- **High:** Address as soon as possible
- **Moderate:** Fix when convenient
- **Low:** Informational

**Alternative:** Use `yarn audit` for Yarn projects.

---

## Quick Reference

### Install All Tools

**Python:**
```bash
uv pip install ruff mypy black isort vulture bandit types-PyYAML
```

**Go:**
```bash
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
go install github.com/securego/gosec/v2/cmd/gosec@latest
```

**Shell:**
```bash
# Ubuntu/Debian
sudo apt install shellcheck
pip install bashate

# macOS
brew install shellcheck
pip install bashate
```

**C++:**
```bash
# Ubuntu/Debian
sudo apt install clang-tidy cppcheck clang-format

# macOS
brew install llvm cppcheck
```

**Java:**
```bash
# Download checkstyle JAR
curl -LO https://github.com/checkstyle/checkstyle/releases/download/checkstyle-10.12.5/checkstyle-10.12.5-all.jar
```

**JavaScript/TypeScript:**
```bash
npm install -g eslint prettier typescript
```

### Verify Installation

Run the codereview CLI with `--static-analysis` flag. Only installed tools will run:

```bash
uv run codereview ./src --static-analysis --dry-run
```

The output will show which tools are available and will be used during analysis.

---

## Implementation notes

Background for the Static-analysis rules in `CLAUDE.md`. Read this before changing
`static_analysis.py`.

### The config-execution gate

`run_tool` consults `_find_executable_config` (against `_CONFIG_EXECUTION_RISK`) **before the
command is built**, and returns a failed result naming the config rather than running. Verified
live against mypy 1.19.1, ESLint v10.8.0 and Prettier 3.x: each ran attacker-supplied code, and
mypy still reported `passed: True`.

Two design rules, both deliberate:

- **Detect on *content*, not presence.** Suffix-executable configs (`eslint.config.js`) are risky
  by existing; data configs are risky only when `_PLUGIN_DECLARATION` matches — and for mypy, only
  inside its own `[mypy]`/`[tool.mypy]` section. Refusing every repo that merely ships a
  `pyproject.toml` would trade away the feature's whole point (linter output that matches *that*
  project's CI) for a threat that isn't there. This repository is asserted not-false-positived by
  `test_this_repository_is_not_false_positived`.
- **Fail closed.** An `OSError` reading the config, or one over `_MAX_CONFIG_SCAN_BYTES` (512 KB),
  counts as risky — an unreadable file is exactly what an attacker would arrange if that bypassed
  the check.

The skip message must say the tool's **findings are missing from this review**; a silently-absent
tool reads as "clean". Neutralizing flags exist (`mypy --config-file=`,
`eslint --no-config-lookup`, `prettier --no-config`) but were not used: they'd silently produce
output that doesn't match the project's own config, which is a different lie.

Locked by the `_CONFIG_EXECUTION_RISK` block in `tests/test_static_analysis.py`, which patches
`codereview.static_analysis.subprocess.run` and asserts `run.assert_not_called()` — asserting on
the *result* would pass even if the tool ran first.

### "I ran and found problems" ≠ "I never started"

`_OPERATIONAL_FAILURE_EXIT_CODES` maps the exit codes that mean a tool *couldn't run*
(ruff/black/mypy: 2) so the result carries `issues_count=0` plus an explicit error, not a
fabricated finding count. Without it, a repo whose ruff config names a nonexistent rule
contributed **zero** coverage while reporting a tidy count.

The map is deliberately short and **verified against the installed binaries, not the docs**:
isort exits 1 for both a bad config and a mis-sorted file, and vulture exits 3 for findings, so
neither is classifiable and neither is listed. When a tool's two meanings are ambiguous, keep
treating the exit as a finding — discarding real findings is the worse error.

### Parse structured output from `stdout` only

`_count_npm_audit_issues` was fed `stdout + stderr`, and npm writes routine notices there
(`npm warn …`), so a single warning turned valid `--json` output into unparseable text and the
count silently became 0 — a clean bill of health for a repo with real advisories. Keep
`result.stderr` for the human-readable `output`/`errors` fields.

### Determinism

When `MAX_FILES_PER_TOOL=500` truncation triggers, file lists must be `sorted(...)[:N]`. Locked in
by `tests/test_static_analysis.py::test_truncation_is_deterministic`. `MAX_FILES_PER_TOOL`'s
docstring documents this as a design guarantee.

### Path filtering for prompt condensation matches whole components, from the right

`_line_mentions_any_path` compares component tuples (`_path_components`) at component granularity;
`_path_match_token` keeps the **whole** normalized path. A two-component `parent/basename` token
cannot disambiguate same-named files under same-named parents — and `api/`, `utils/`, `models/`,
`tests/` are precisely the directory names that repeat — so a finding in `other/api/views.py` was
attributed to `app/api/views.py`. Boundary-aware substring matching fixes the
`foo.py` ⊂ `foo.py.orig` shape but *not* this one: the information was already discarded when the
token was built.

Don't reintroduce a fixed-width token, and don't `lstrip("./")` (it eats a leading `.` from a real
name and regresses `test_condense_for_prompt_filter_matches_across_path_forms`).
