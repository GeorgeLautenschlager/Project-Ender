#!/usr/bin/env bash
# Label the full corpus with DeepSeek R1 8B via Ollama.
# Estimated time: ~11h for 10k states on GTX 1070.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== DeepSeek R1:8b labelling pass ==="
echo "Started: $(date)"
echo ""

poetry run python -m project_ender.labeller \
    --db src/corpus.db \
    --backend "ollama:deepseek-r1:8b" \
    --retries 3

echo ""
echo "Finished: $(date)"
