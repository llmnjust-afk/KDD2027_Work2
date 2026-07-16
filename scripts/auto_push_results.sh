#!/usr/bin/env bash
# Auto-push experiment results to GitHub after experiment completes.
# Reads GH_TOKEN from environment (configured in ~/.bashrc), never hardcodes it.
# Usage: ./scripts/auto_push_results.sh <results_dir> [commit_message]
#
# This script is safe to commit to git: it contains no secrets.

set -euo pipefail

RESULTS_DIR="${1:-results_real}"
COMMIT_MSG="${2:-Update experiment results}"

# load env
source ~/.bashrc 2>/dev/null || true

if [ -z "${GH_TOKEN:-}" ]; then
    echo "ERROR: GH_TOKEN not set. Configure it in ~/.bashrc first." >&2
    exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

echo "=== Auto-Push Results to GitHub ==="
echo "Results dir: $RESULTS_DIR"
echo "Repo: $REPO_DIR"

# ensure git identity is set
git config user.email "agentfail@kdd2027.local" 2>/dev/null || true
git config user.name "AgentFail" 2>/dev/null || true

# stage results (force add to bypass .gitignore if needed)
git add -f "$RESULTS_DIR" 2>/dev/null || {
    echo "WARNING: could not add $RESULTS_DIR (does it exist?)"
    ls -la "$RESULTS_DIR" 2>/dev/null || echo "  dir not found"
}

# also stage any new experiment scripts/logs if present
git add -A 2>/dev/null || true

# check if there's anything to commit
if git diff --cached --quiet 2>/dev/null; then
    echo "Nothing new to commit."
    exit 0
fi

# commit
git commit -m "$COMMIT_MSG" 2>&1 | tail -3

# push using token in URL (temporary, not stored in git config)
REMOTE_URL="https://x-access-token:${GH_TOKEN}@github.com/llmnjust-afk/KDD2027_Work2.git"
git -c credential.helper= push "$REMOTE_URL" main 2>&1 | tail -5

echo "=== Push complete ==="
echo "Results available at: https://github.com/llmnjust-afk/KDD2027_Work2"
