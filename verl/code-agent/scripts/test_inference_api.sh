#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:8080}"

curl -sS "${API_URL}/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Write a Python function add(a, b) that returns the sum.",
    "max_tokens": 512,
    "temperature": 0.2,
    "top_p": 0.95
  }'
echo
