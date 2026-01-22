# Usage Guide

Comprehensive guide for using the Code Review CLI tool effectively.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Typical Workflows](#typical-workflows)
3. [Use Cases](#use-cases)
4. [Best Practices](#best-practices)
5. [Configuration Options](#configuration-options)
6. [Understanding Results](#understanding-results)
7. [Performance Tuning](#performance-tuning)
8. [Advanced Usage](#advanced-usage)

## Getting Started

### First Time Setup

1. **Install the tool**:
   ```bash
   uv pip install -e .
   ```

2. **Configure AWS credentials**:
   ```bash
   aws configure
   ```
   Enter your AWS Access Key ID, Secret Access Key, and preferred region (recommend us-west-2).

3. **Verify setup**:
   ```bash
   codereview --help
   ```

4. **Run your first review**:
   ```bash
   codereview ./src --verbose
   ```

### Quick Start Checklist

- [ ] AWS credentials configured
- [ ] Bedrock access enabled in AWS Console
- [ ] Claude Opus 4.5 model access approved
- [ ] IAM permissions include `bedrock:InvokeModel`
- [ ] Tool installed and `codereview` command available

## Typical Workflows

### Workflow 1: Pre-Commit Review

Review changes before committing to version control:

```bash
# Review only changed files (staged for commit)
git diff --name-only --cached | xargs dirname | sort -u | head -1 | xargs codereview

# Or review a specific feature branch directory
codereview ./src/features/new-feature --severity medium
```

**When to use**: Before committing significant changes, especially for critical features.

### Workflow 2: Pull Request Review

Generate a comprehensive review for PR description:

```bash
# Review the branch and export to markdown
codereview ./src --output pr-review.md --severity high

# Share pr-review.md in your PR description or comments
```

**When to use**: When creating PRs, especially for complex features or refactoring.

### Workflow 3: Legacy Code Audit

Analyze existing codebase for technical debt:

```bash
# Full review with all severity levels
codereview ./legacy-module --output audit-report.md

# Focus on security and performance
codereview ./legacy-module --severity medium --output security-audit.md
```

**When to use**: When planning refactoring efforts or security audits.

### Workflow 4: New Developer Onboarding

Help new team members understand code quality expectations:

```bash
# Review their first contribution
codereview ./their-feature --output onboarding-feedback.md --verbose
```

**When to use**: During code review for new team members' first PRs.

### Workflow 5: CI/CD Integration

Automate code reviews in your pipeline:

```bash
# In your CI/CD script
codereview ./src \
  --output review-report.md \
  --severity high \
  --max-files 100

# Fail pipeline if critical issues found
if grep -q "Critical" review-report.md; then
  echo "Critical issues found!"
  exit 1
fi
```

**When to use**: As part of automated quality gates in CI/CD.

## Use Cases

### Use Case 1: Security Audit

**Goal**: Identify security vulnerabilities in authentication module.

```bash
codereview ./src/auth \
  --output security-audit.md \
  --severity medium \
  --verbose
```

**What to look for**:
- SQL injection vulnerabilities
- Authentication bypass risks
- Credential exposure
- Input validation issues

### Use Case 2: Performance Optimization

**Goal**: Find performance bottlenecks in data processing pipeline.

```bash
codereview ./src/pipeline \
  --output performance-review.md \
  --max-files 50
```

**What to look for**:
- Inefficient algorithms (O(n²) loops)
- Database query optimization
- Memory leaks
- Unnecessary computations

### Use Case 3: Code Quality Improvement

**Goal**: Reduce technical debt in core business logic.

```bash
codereview ./src/business \
  --output quality-report.md \
  --severity low
```

**What to look for**:
- Code duplication
- Complex functions needing refactoring
- Missing error handling
- Poor naming conventions

### Use Case 4: API Design Review

**Goal**: Validate REST API design and implementation.

```bash
codereview ./src/api \
  --output api-review.md \
  --severity medium
```

**What to look for**:
- RESTful design principles
- Error response handling
- Input validation
- API versioning approach

### Use Case 5: Microservice Architecture Review

**Goal**: Review service boundaries and inter-service communication.

```bash
codereview ./services \
  --output architecture-review.md \
  --max-file-size 20
```

**What to look for**:
- Service coupling issues
- Communication patterns
- Error propagation
- Configuration management

## Best Practices

### 1. Start Small

Begin with a small directory to understand the output:

```bash
codereview ./src/utils --verbose
```

Gradually expand to larger codebases once familiar.

### 2. Use Appropriate Severity Filters

**For daily development**:
```bash
codereview ./src --severity medium
```

**For production deployments**:
```bash
codereview ./src --severity high
```

**For learning and improvement**:
```bash
codereview ./src --severity info
```

### 3. Leverage Exclusion Patterns

Exclude irrelevant files to focus analysis:

```bash
codereview ./src \
  --exclude "**/tests/**" \
  --exclude "**/migrations/**" \
  --exclude "**/*_pb2.py"
```

### 4. Set File Size Limits

Avoid analyzing generated or minified files:

```bash
codereview ./src --max-file-size 15
```

Typical limits:
- Small projects: 10KB (default)
- Medium projects: 15KB
- Large projects: 20KB

### 5. Use Markdown Export for Sharing

Always export important reviews:

```bash
codereview ./src --output review-$(date +%Y%m%d).md
```

Benefits:
- Version control for reviews
- Easy sharing with team
- Historical reference

### 6. Run Regular Reviews

Establish a cadence:
- **Daily**: Review your own changes
- **Weekly**: Team code quality review
- **Monthly**: Full codebase audit
- **Quarterly**: Architecture review

### 7. Combine with Git

Review only modified files:

```bash
# Review files changed in last commit
git diff --name-only HEAD~1 | xargs -I {} dirname {} | sort -u | xargs codereview

# Review files in current branch vs main
git diff main --name-only | xargs -I {} dirname {} | sort -u | xargs codereview
```

### 8. Use Verbose Mode for Debugging

When issues occur:

```bash
codereview ./src --verbose
```

Provides:
- Detailed progress information
- Full error traces
- AWS API call details

### 9. Monitor Token Usage

Be aware of costs:
- Claude Opus 4.5 is powerful but expensive
- Use `--max-files` to limit scope
- Focus on critical paths first

### 10. Act on Findings Systematically

Don't try to fix everything at once:

1. **Immediate**: Fix Critical and High severity issues
2. **Short-term**: Address Medium severity issues
3. **Long-term**: Plan for Low severity improvements
4. **Education**: Learn from Info-level insights

## Configuration Options

### File Size Management

```bash
# Default: 10KB
codereview ./src

# Smaller files only
codereview ./src --max-file-size 5

# Larger files included
codereview ./src --max-file-size 20
```

### File Count Limits

```bash
# Analyze only first 50 files
codereview ./src --max-files 50

# Useful for:
# - Quick checks
# - Cost control
# - Testing configuration
```

### Region Selection

```bash
# Default: us-west-2
codereview ./src

# Specific region
codereview ./src --aws-region us-east-1

# Reasons to change region:
# - Model availability
# - Latency optimization
# - Compliance requirements
```

### AWS Profile

```bash
# Use specific AWS profile
codereview ./src --aws-profile production

# Useful for:
# - Multiple AWS accounts
# - Different IAM roles
# - Environment separation
```

## Understanding Results

### Interpreting Issues

Each issue includes:

**Category**: Type of problem
- Focus on Security and Performance first
- Code Quality issues indicate technical debt
- System Design insights are architectural

**Severity**: Priority level
- Critical: Fix immediately
- High: Fix before deployment
- Medium: Address in current sprint
- Low: Plan for future
- Info: Consider for improvement

**File Path & Line Numbers**: Exact location
- Navigate directly to the issue
- Review context around the lines

**Description**: What's wrong
- Read carefully to understand the problem
- Check if it's a false positive

**Suggested Code**: How to fix it
- Review the suggestion
- Adapt to your context
- Don't blindly copy-paste

**Rationale**: Why it matters
- Understand the underlying principle
- Learn for future development

**References**: Learn more
- Follow links for deep dives
- Share with team for education

### Reading the Summary

The summary provides:
- Total files analyzed
- Issue count by severity
- Overall health assessment

Use it to:
- Track progress over time
- Compare modules/features
- Set improvement goals

### System Design Insights

These are high-level observations about:
- Architecture patterns
- Service boundaries
- Scalability concerns
- Maintainability issues

Don't ignore these - they often reveal systemic issues.

### Recommendations

Priority actions based on analysis:
- Usually 3-5 top items
- Ordered by impact
- Actionable next steps

## Performance Tuning

### Reducing Analysis Time

**Strategy 1: Limit files**
```bash
codereview ./src --max-files 50
```

**Strategy 2: Increase file size limit (fewer files)**
```bash
codereview ./src --max-file-size 20
```

**Strategy 3: Exclude large directories**
```bash
codereview ./src --exclude "**/tests/**" --exclude "**/vendor/**"
```

### Handling Large Codebases

For projects with 1000+ files:

1. **Divide and conquer**:
   ```bash
   codereview ./src/core --output core-review.md
   codereview ./src/api --output api-review.md
   codereview ./src/utils --output utils-review.md
   ```

2. **Focus on critical paths**:
   ```bash
   codereview ./src/auth ./src/payment --output critical-review.md
   ```

3. **Use max-files strategically**:
   ```bash
   # First 100 files
   codereview ./src --max-files 100 --output batch1.md
   ```

### Managing AWS Costs

Claude Opus 4.5 pricing considerations:

1. **Estimate tokens**: ~100 tokens per 4 lines of code
2. **Use max-files**: Limit scope for cost control
3. **Review incrementally**: Analyze changes, not entire codebase
4. **Filter by severity**: Reduce output size with severity filters

Example cost-conscious workflow:
```bash
# Focus on high-priority issues only
codereview ./src \
  --severity high \
  --max-files 50 \
  --max-file-size 10
```

## Advanced Usage

### Custom Exclusion Patterns

```bash
codereview ./src \
  --exclude "**/*_test.go" \
  --exclude "**/*.min.js" \
  --exclude "**/node_modules/**" \
  --exclude "**/vendor/**"
```

### Combining Multiple Options

```bash
codereview ./src \
  --output detailed-review.md \
  --severity medium \
  --max-files 200 \
  --max-file-size 15 \
  --exclude "**/tests/**" \
  --aws-region us-west-2 \
  --verbose
```

### Scripting Reviews

Create a review script `review.sh`:

```bash
#!/bin/bash
DATE=$(date +%Y%m%d)
OUTPUT_DIR="./reviews"

mkdir -p "$OUTPUT_DIR"

echo "Running code review..."
codereview ./src \
  --output "$OUTPUT_DIR/review-$DATE.md" \
  --severity high \
  --verbose

echo "Review complete: $OUTPUT_DIR/review-$DATE.md"
```

### Environment-Specific Reviews

Development:
```bash
codereview ./src --severity info --verbose
```

Staging:
```bash
codereview ./src --severity medium --output staging-review.md
```

Production:
```bash
codereview ./src --severity high --output prod-review.md --aws-region us-west-2
```

### Integration with Git Hooks

Pre-commit hook (`.git/hooks/pre-commit`):

```bash
#!/bin/bash
# Run code review on staged files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(py|go)$')

if [ -n "$STAGED_FILES" ]; then
  echo "Running code review on staged files..."
  codereview ./src --severity high --max-files 20
fi
```

### Continuous Monitoring

Set up weekly reviews via cron:

```bash
# Run every Monday at 9 AM
0 9 * * 1 cd /path/to/project && codereview ./src --output weekly-review-$(date +\%Y\%m\%d).md
```

## Tips and Tricks

1. **Quick health check**:
   ```bash
   codereview ./src --severity critical
   ```
   Zero critical issues = good to go.

2. **Focus on new code**:
   ```bash
   codereview ./src/new-feature --max-files 30
   ```

3. **Compare before and after refactoring**:
   ```bash
   codereview ./src --output before-refactor.md
   # ... do refactoring ...
   codereview ./src --output after-refactor.md
   diff before-refactor.md after-refactor.md
   ```

4. **Learn from issues**:
   Read Info-level issues to improve coding skills.

5. **Share best examples**:
   Export reviews of good code as examples for the team.

## Getting Help

If you encounter issues:

1. **Check verbose output**:
   ```bash
   codereview ./src --verbose
   ```

2. **Verify AWS setup**:
   ```bash
   aws bedrock list-foundation-models --region us-west-2
   ```

3. **Test with small directory**:
   ```bash
   codereview ./src/single-file-dir --verbose
   ```

4. **Review troubleshooting guide** in README.md

5. **Check AWS Bedrock quotas** in AWS Console
