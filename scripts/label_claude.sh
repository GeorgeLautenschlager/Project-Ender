#!/usr/bin/env bash
# Label the corpus with Claude via CLI until usage limit is hit.
# Exits gracefully on usage limit — safe to run unattended.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Claude labelling pass (stops at usage limit) ==="
echo "Started: $(date)"
echo ""

poetry run python -m project_ender.labeller \
    --db src/corpus.db \
    --backend claude \
    --retries 2 \
    --stop-on-usage-limit

echo ""
echo "Finished: $(date)"
