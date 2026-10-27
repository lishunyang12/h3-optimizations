#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run the deployed 5s avatar and 15s FastH3 reference on the target system.
# This name is a handoff entry point, not a claim of validated PRO 5500 support.
# The shared launcher checks the actual GPU identity, count, architecture and RAM.
set -euo pipefail
h3_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$h3_repo_root/scripts/sku_benchmark.py" suite --cases main "$@"
