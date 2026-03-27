#!/usr/bin/env bash
# Label the corpus with a model running in LM Studio over Tailscale.
#
# Usage:
#   ./scripts/label_lmstudio.sh                           # defaults
#   ./scripts/label_lmstudio.sh 100.64.0.5:1234 gpt-oss-20b
#
# Args:
#   $1  LM Studio host:port  (default: localhost:1234)
#   $2  Model name            (default: gpt-oss-20b)
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${1:-localhost:1234}"
MODEL="${2:-gpt-oss-20b}"
BACKEND="lmstudio:${MODEL}@${HOST}"

echo "=== LM Studio labelling pass ==="
echo "Backend: ${BACKEND}"
echo "Started: $(date)"
echo ""

poetry run python -m project_ender.labeller \
    --db src/corpus.db \
    --backend "${BACKEND}" \
    --retries 3

echo ""
echo "Finished: $(date)"
