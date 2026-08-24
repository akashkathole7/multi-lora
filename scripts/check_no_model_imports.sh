#!/usr/bin/env bash
# Guardrail: the verifier and the eval harness must stay model-free.
#
# data/verifier.py is the data filter, the training gate and the final proof.
# eval/ is where results get measured. If either one can call a model, the
# numbers they produce stop being evidence. This script fails the build if a
# model-client import appears in those paths.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TARGETS=("data/verifier.py" "eval")
PATTERN='^[[:space:]]*(import|from)[[:space:]]+(anthropic|openai|google\.generativeai|genai|mistralai|cohere|ollama|litellm|langchain)'

echo "guardrail: checking ${TARGETS[*]} for model-client imports"

hits=""
for target in "${TARGETS[@]}"; do
  if [ ! -e "$target" ]; then
    echo "guardrail: target $target does not exist"
    exit 1
  fi
  found="$(grep -rInE "$PATTERN" "$target" || true)"
  if [ -n "$found" ]; then
    hits="${hits}${found}"$'\n'
  fi
done

if [ -n "$hits" ]; then
  echo "guardrail: FAIL - model-client import found in a model-free path:"
  echo "$hits"
  echo "guardrail: data/verifier.py and eval/ must not import a model client."
  exit 1
fi

echo "guardrail: PASS - no model-client imports in ${TARGETS[*]}"
exit 0
