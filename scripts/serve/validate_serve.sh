#!/bin/bash
# Check that a code or feedback server completes a generation request.
# Usage: bash scripts/serve/validate_serve.sh http://host:8000/v1 MODEL_NAME
set -euo pipefail
SERVE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="${1:?usage: validate_serve.sh BASE_URL MODEL_NAME}"
SERVED_NAME="${2:?usage: validate_serve.sh BASE_URL MODEL_NAME}"
BASE_URL="${BASE_URL%/}"
HEADERS=(-H 'Content-Type: application/json')
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  HEADERS+=(-H "Authorization: Bearer $OPENAI_API_KEY")
fi
PAYLOAD="$(python3 - "$SERVED_NAME" <<'JSON'
import json, sys
print(json.dumps({"model": sys.argv[1], "messages": [
    {"role": "user", "content": "Reply with exactly: OK"}
], "max_tokens": 800}))
JSON
)"
curl --fail --silent --show-error --max-time 300 \
  "$BASE_URL/chat/completions" "${HEADERS[@]}" --data "$PAYLOAD" \
  | SN="$SERVED_NAME" ENDPOINT="$BASE_URL" \
    python3 "$SERVE_DIR/_validate_serve_parse.py"
