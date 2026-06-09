#!/usr/bin/env bash
# Lint + format check (matches CI). Pass --fix to auto-apply.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGETS=(src experiments tests)

if [[ "${1:-}" == "--fix" ]]; then
    ruff check --fix "${TARGETS[@]}"
    black "${TARGETS[@]}"
else
    ruff check "${TARGETS[@]}"
    black --check "${TARGETS[@]}"
fi
