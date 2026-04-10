#!/bin/bash
# Run DISCOVA vs Vanilla comparison for all pipelines sequentially.
# Results are cached per pipeline, so re-running is safe and resumes from where it left off.
set -e
cd /home/abhireddy/Desktop/inlp/PROJ

echo "=== DISCOVA vs Vanilla — Full Comparison Run ==="
echo "Pipelines: A (main), B, C, D (ablations)"
echo ""

python3 -m src.compare_discova_vanilla --pipelines A B C D --top-n 5 2>&1 | tee data/comparison_run.log

echo ""
echo "=== Done. Check data/comparison/ for global results ==="
