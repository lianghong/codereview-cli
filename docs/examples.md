# Examples

Real-world examples and CI/CD integration patterns for the Code Review CLI tool.

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Advanced Examples](#advanced-examples)
3. [CI/CD Integration](#cicd-integration)
4. [Example Output](#example-output)
5. [Real-World Scenarios](#real-world-scenarios)

## Basic Examples

### Example 1: Quick Review

Review a small feature with default settings:

```bash
codereview ./src/features/user-auth
```

**Output**:
```
🔍 Code Review Tool

📂 Scanning directory: ./src/features/user-auth

✓ Found 8 files to review

📦 Created 1 batches

Analyzing code... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

📊 Code Review Report
═══════════════════════════════════════════════

Summary: Analyzed 8 files and found 12 issues

Critical: 1 | High: 3 | Medium: 5 | Low: 2 | Info: 1
```

### Example 2: Export to Markdown

Generate a shareable report:

```bash
codereview ./src/api --output api-review.md
```

**Creates**: `api-review.md` with full review details.

### Example 3: Filter by Severity

Show only critical and high-severity issues:

```bash
codereview ./src --severity high
```

**Result**: Displays only Critical and High issues, hiding Medium/Low/Info.

### Example 4: Exclude Test Files

Skip test files during review:

```bash
codereview ./src --exclude "**/tests/**" --exclude "**/*_test.py"
```

### Example 5: Limit Analysis Scope

Analyze only the first 50 files:

```bash
codereview ./large-project --max-files 50
```

### Example 6: Model Selection

Choose the right model for your use case:

```bash
# High-quality review with Opus (critical code)
codereview ./src/auth --model opus

# Balanced review with Sonnet (daily development)
codereview ./src --model sonnet

# Fast review with Haiku (large codebase)
codereview ./monorepo --model haiku --max-files 1000
```

**Output with Model Information**:
```
🔍 Code Review Tool

📂 Scanning directory: ./src
🤖 Model: Claude Sonnet 4.5

✓ Found 50 files to review

💰 Token Usage & Cost Estimate:
   Input tokens:  25,000
   Output tokens: 8,000
   Total tokens:  33,000
   Estimated cost: $0.1950
```

## Advanced Examples

### Example 7: Multi-Module Review

Review multiple modules separately:

```bash
#!/bin/bash
MODULES=("auth" "api" "database" "utils")
OUTPUT_DIR="./reviews/$(date +%Y%m%d)"

mkdir -p "$OUTPUT_DIR"

for module in "${MODULES[@]}"; do
  echo "Reviewing module: $module"
  codereview "./src/$module" \
    --output "$OUTPUT_DIR/$module-review.md" \
    --severity medium
done

echo "All reviews complete in $OUTPUT_DIR"
```

### Example 8: Incremental Review Script

Review only changed files since last commit:

```bash
#!/bin/bash
# review-changes.sh

# Get list of changed Python and Go files
CHANGED_FILES=$(git diff --name-only HEAD~1 | grep -E '\.(py|go)$')

if [ -z "$CHANGED_FILES" ]; then
  echo "No Python or Go files changed"
  exit 0
fi

# Get unique directories
DIRS=$(echo "$CHANGED_FILES" | xargs -I {} dirname {} | sort -u)

# Review each directory
for dir in $DIRS; do
  echo "Reviewing: $dir"
  codereview "$dir" --severity high
done
```

### Example 9: Cost-Conscious Review with Model Selection

Minimize AWS costs while maintaining quality by choosing the right model:

```bash
# Option 1: Sonnet for balanced cost/quality
codereview ./src/core ./src/security \
  --model sonnet \
  --severity high \
  --max-files 100 \
  --max-file-size 10 \
  --output critical-review.md

# Option 2: Haiku for large scans
codereview ./src \
  --model haiku \
  --max-files 500 \
  --severity high \
  --output large-scan.md
```

**Cost Comparison**:
- Opus (100 files): ~$1.50
- Sonnet (100 files): ~$0.30
- Haiku (500 files): ~$0.20

### Example 10: Region-Specific Configuration

Use different AWS regions for redundancy:

```bash
# Primary region
codereview ./src --aws-region us-west-2 --output review-west.md || \
# Fallback region
codereview ./src --aws-region us-east-1 --output review-east.md
```

### Example 11: Verbose Debugging

Debug issues with detailed output:

```bash
codereview ./problematic-module \
  --verbose \
  --max-files 10 \
  --output debug-review.md 2>&1 | tee debug.log
```

## CI/CD Integration

**Cost-Effective CI/CD**: Use Haiku or Sonnet in CI/CD to reduce costs while maintaining quality gates.

### GitHub Actions

**`.github/workflows/code-review.yml`**:

```yaml
name: AI Code Review

on:
  pull_request:
    types: [opened, synchronize]
  push:
    branches: [main, develop]

jobs:
  code-review:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.14'

      - name: Install uv
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          echo "$HOME/.cargo/bin" >> $GITHUB_PATH

      - name: Install code review tool
        run: |
          uv venv --python 3.14
          uv pip install -e .

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-west-2

      - name: Run code review
        run: |
          codereview ./src \
            --model sonnet \
            --output review-report.md \
            --severity high \
            --max-files 100

      - name: Upload review report
        uses: actions/upload-artifact@v3
        with:
          name: code-review-report
          path: review-report.md

      - name: Check for critical issues
        run: |
          if grep -q "Critical" review-report.md; then
            echo "::error::Critical issues found in code review"
            exit 1
          fi

      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('review-report.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## AI Code Review Results\n\n${report}`
            });
```

### GitLab CI

**`.gitlab-ci.yml`**:

```yaml
stages:
  - review

code-review:
  stage: review
  image: python:3.14
  before_script:
    - curl -LsSf https://astral.sh/uv/install.sh | sh
    - export PATH="$HOME/.cargo/bin:$PATH"
    - uv venv --python 3.14
    - uv pip install -e .
  script:
    - |
      codereview ./src \
        --model haiku \
        --output review-report.md \
        --severity high \
        --max-files 100
    - |
      if grep -q "Critical" review-report.md; then
        echo "Critical issues found!"
        cat review-report.md
        exit 1
      fi
  artifacts:
    paths:
      - review-report.md
    expire_in: 1 week
  only:
    - merge_requests
    - main
  variables:
    AWS_ACCESS_KEY_ID: $AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY: $AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION: us-west-2
```

### Jenkins Pipeline

**`Jenkinsfile`**:

```groovy
pipeline {
    agent any

    environment {
        AWS_CREDENTIALS = credentials('aws-bedrock-credentials')
    }

    stages {
        stage('Setup') {
            steps {
                sh '''
                    curl -LsSf https://astral.sh/uv/install.sh | sh
                    export PATH="$HOME/.cargo/bin:$PATH"
                    uv venv --python 3.14
                    uv pip install -e .
                '''
            }
        }

        stage('Code Review') {
            steps {
                withCredentials([
                    string(credentialsId: 'aws-access-key', variable: 'AWS_ACCESS_KEY_ID'),
                    string(credentialsId: 'aws-secret-key', variable: 'AWS_SECRET_ACCESS_KEY')
                ]) {
                    sh '''
                        codereview ./src \
                            --model haiku \
                            --output review-report.md \
                            --severity high \
                            --max-files 100 \
                            --aws-region us-west-2
                    '''
                }
            }
        }

        stage('Check Results') {
            steps {
                script {
                    def report = readFile('review-report.md')
                    if (report.contains('Critical')) {
                        error('Critical issues found in code review')
                    }
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'review-report.md', fingerprint: true
        }
        failure {
            emailext(
                subject: "Code Review Failed: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                body: "Critical issues found. See attached report.",
                attachmentsPattern: 'review-report.md',
                to: '${DEFAULT_RECIPIENTS}'
            )
        }
    }
}
```

### CircleCI

**`.circleci/config.yml`**:

```yaml
version: 2.1

jobs:
  code-review:
    docker:
      - image: cimg/python:3.14
    steps:
      - checkout

      - run:
          name: Install uv
          command: |
            curl -LsSf https://astral.sh/uv/install.sh | sh
            echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> $BASH_ENV

      - run:
          name: Install dependencies
          command: |
            uv venv --python 3.14
            uv pip install -e .

      - run:
          name: Run code review
          command: |
            codereview ./src \
              --output review-report.md \
              --severity high \
              --max-files 100
          environment:
            AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
            AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
            AWS_DEFAULT_REGION: us-west-2

      - run:
          name: Check for critical issues
          command: |
            if grep -q "Critical" review-report.md; then
              echo "Critical issues found!"
              exit 1
            fi

      - store_artifacts:
          path: review-report.md
          destination: code-review

workflows:
  version: 2
  review:
    jobs:
      - code-review:
          filters:
            branches:
              only:
                - main
                - develop
```

### Pre-Commit Hook

**`.git/hooks/pre-commit`**:

```bash
#!/bin/bash
# Pre-commit code review hook

echo "Running AI code review on staged files..."

# Get staged Python and Go files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(py|go)$')

if [ -z "$STAGED_FILES" ]; then
  echo "No Python or Go files staged for commit"
  exit 0
fi

# Get unique directories
DIRS=$(echo "$STAGED_FILES" | xargs -I {} dirname {} | sort -u)

# Review directories with staged changes
for dir in $DIRS; do
  echo "Reviewing: $dir"
  codereview "$dir" --severity high --max-files 20

  if [ $? -ne 0 ]; then
    echo "❌ Code review found issues. Please fix before committing."
    exit 1
  fi
done

echo "✅ Code review passed"
exit 0
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

## Example Output

### Terminal Output

```
🔍 Code Review Tool

📂 Scanning directory: ./src/auth

✓ Found 12 files to review

📦 Created 2 batches

Analyzing code... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

📊 Code Review Report
═══════════════════════════════════════════════

Summary: Analyzed 12 files and found 8 issues

Critical: 1 | High: 2 | Medium: 3 | Low: 1 | Info: 1

🚨 Critical Issues (1)
─────────────────────────────────────────────

[Security] SQL Injection Vulnerability
File: src/auth/login.py:45-47
Severity: Critical

User input is directly interpolated into SQL query without
sanitization, allowing SQL injection attacks.

Rationale: Attackers can inject malicious SQL to bypass
authentication or access sensitive data.

Suggested Fix:
  # Use parameterized queries
  cursor.execute(
      "SELECT * FROM users WHERE username = %s AND password = %s",
      (username, hashed_password)
  )

References:
  - https://owasp.org/www-community/attacks/SQL_Injection

⚠️  High Priority Issues (2)
─────────────────────────────────────────────

[Best Practices] Missing Error Handling
File: src/auth/session.py:78
Severity: High

Database operations lack try-catch blocks, potentially causing
unhandled exceptions.

[Performance] Inefficient Query in Loop
File: src/auth/permissions.py:112-125
Severity: High

N+1 query problem: executing database query inside loop instead
of batch loading.

💡 System Design Insights
─────────────────────────────────────────────

The authentication module shows tight coupling between session
management and database access. Consider:
- Implementing a repository pattern
- Separating business logic from data access
- Adding caching layer for frequently accessed permissions

📋 Top Recommendations
─────────────────────────────────────────────

1. 🚨 Fix SQL injection vulnerability immediately
2. ⚠️  Add comprehensive error handling
3. 📊 Optimize database query patterns
```

### Markdown Report Example

**`review-report.md`**:

```markdown
# Code Review Report

**Generated**: 2026-01-22
**Directory**: ./src/auth
**Files Analyzed**: 12
**Total Issues**: 8

## Summary

Analyzed 12 files and found 8 issues across multiple categories.

## Metrics

| Metric | Count |
|--------|-------|
| Files Analyzed | 12 |
| Total Issues | 8 |
| Critical | 1 |
| High | 2 |
| Medium | 3 |
| Low | 1 |
| Info | 1 |

## Critical Issues

### SQL Injection Vulnerability
- **File**: `src/auth/login.py:45-47`
- **Category**: Security
- **Severity**: Critical

User input is directly interpolated into SQL query...

**Suggested Fix**:
```python
# Use parameterized queries
cursor.execute(
    "SELECT * FROM users WHERE username = %s",
    (username,)
)
```

## System Design Insights

The authentication module shows tight coupling...

## Recommendations

1. Fix SQL injection vulnerability immediately
2. Add comprehensive error handling
3. Optimize database query patterns
```

## Real-World Scenarios

### Scenario 1: New Microservice Review

**Context**: Team built a new payment processing microservice.

**Command**:
```bash
codereview ./services/payment \
  --output payment-service-review.md \
  --severity medium \
  --verbose
```

**Action Items from Review**:
1. Add input validation for payment amounts
2. Implement idempotency keys
3. Add proper error handling for external API calls
4. Review transaction rollback logic

### Scenario 2: Legacy Code Modernization

**Context**: Planning to modernize 5-year-old Python 2.7 codebase.

**Command**:
```bash
codereview ./legacy-app \
  --output modernization-audit.md \
  --severity low \
  --max-file-size 20
```

**Findings**:
- 45 High-severity issues (mostly security)
- 120 Medium-severity issues (code quality)
- Architecture needs significant refactoring
- Estimate: 3 sprints for critical fixes

### Scenario 3: Security Audit Before Launch

**Context**: Pre-production security review for e-commerce platform.

**Command**:
```bash
codereview ./src \
  --output security-audit-2026-01-22.md \
  --severity high \
  --exclude "**/tests/**"
```

**Findings**:
- 2 Critical: SQL injection, XSS vulnerability
- 5 High: Authentication issues, insecure deserialization
- Blocked deployment until Critical issues fixed

### Scenario 4: Code Quality Gate in CI

**Context**: Automated quality gate to prevent technical debt.

**GitHub Action**:
```yaml
- name: Quality Gate
  run: |
    codereview ./src --output review.md --severity high
    CRITICAL=$(grep -c "Critical" review.md || true)
    if [ "$CRITICAL" -gt 0 ]; then
      echo "❌ $CRITICAL critical issues found"
      exit 1
    fi
```

**Impact**:
- Prevented 3 security vulnerabilities from reaching production
- Reduced post-deployment bugs by 40%
- Improved code review turnaround time

### Scenario 5: Refactoring Validation

**Context**: Validating refactoring reduced complexity.

**Commands**:
```bash
# Before refactoring
codereview ./src/complex-module --output before.md

# After refactoring
codereview ./src/complex-module --output after.md

# Compare
diff before.md after.md
```

**Results**:
- Before: 25 Medium issues (complexity)
- After: 8 Medium issues
- 68% reduction in code quality issues

## Tips for CI/CD Integration

1. **Choose Cost-Effective Models**:
   - **Haiku** for CI/CD quality gates (fastest, cheapest)
   - **Sonnet** for PR reviews (balanced)
   - **Opus** for production releases only (highest quality)

2. **Use Artifacts**: Always save review reports as build artifacts

3. **Set Severity Thresholds**: Fail builds only on Critical/High issues

4. **Limit Scope**: Use `--max-files` to control costs and time

5. **Cache Dependencies**: Cache the tool installation for faster builds

6. **Parallel Reviews**: Review different modules in parallel jobs

7. **Scheduled Reviews**: Run full reviews nightly, not on every commit

8. **Notification Integration**: Send review summaries to Slack/email

9. **Store Metrics**: Track issue trends over time in dashboards

10. **Cost Monitoring**: Track token usage and costs across builds

**Cost Optimization Example**:
```yaml
# Use Haiku for pull requests (fast + cheap)
- name: PR Review
  run: codereview ./src --model haiku

# Use Opus for main branch only (quality)
- name: Production Review
  if: github.ref == 'refs/heads/main'
  run: codereview ./src --model opus
```

## Troubleshooting CI/CD

### Issue: Timeout in CI

**Solution**:
```bash
codereview ./src --max-files 100 --max-file-size 10
```

### Issue: AWS Credentials in CI

**Solution**: Use CI platform's secret management:
- GitHub: Repository Secrets
- GitLab: CI/CD Variables
- Jenkins: Credentials Plugin
- CircleCI: Environment Variables

### Issue: Inconsistent Results

**Solution**: Pin to specific AWS region:
```bash
codereview ./src --aws-region us-west-2
```

### Issue: High Costs

**Solution**: Review only changed files:
```bash
git diff --name-only origin/main | xargs dirname | xargs codereview
```
