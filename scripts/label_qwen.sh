#!/usr/bin/env bash
# Label the full corpus with Qwen 2.5 7B via Ollama.
# Estimated time: ~5h for 10k states on GTX 1070.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Qwen 2.5:7b labelling pass ==="
echo "Started: $(date)"
echo ""

poetry run python -m project_ender.labeller \
    --db src/corpus.db \
    --backend "ollama:qwen2.5:7b" \
    --retries 3

echo ""
echo "Finished: $(date)"
