#!/usr/bin/env bash

set -euo pipefail

# Regenerate the README chart from the checked-in RTX 5060 Ti baseline data.
python3 plot_benchmark_results.py "$@"
