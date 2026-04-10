#!/bin/bash
PROJ="/home/abhireddy/Desktop/inlp/PROJ"
VENV="/home/abhireddy/Desktop/inlp/venv/bin/activate"
cd "$PROJ"
source "$VENV"

echo "[chain] Waiting for Stage 2 to finish..."
while pgrep -f "run_pipeline.py" > /dev/null 2>&1; do sleep 10; done

echo "[chain] Stage 2 done. Quality check..."
python3 -c "
import json
with open('data/stage2_output/distilled_logic.json') as f:
    d = json.load(f)
real = sum(1 for v in d.values() if any(w in v for w in ['Parameter','System Y','optimize','minimize','constrain','converge']))
print(f'  {len(d)} total | {real} real distillations | {len(d)-real} fallback')
"

echo "[chain] Running Stage 3..."
python3 run_pipeline.py --stages 3 2>&1 | tee -a pipeline.log
echo "[chain] Stage 3 done."

echo "[chain] Running Stage 4..."
python3 run_pipeline.py --stages 4 2>&1 | tee -a pipeline.log
echo "[chain] Stage 4 done."

echo "[chain] Running Stage 5..."
python3 run_pipeline.py --stages 5 2>&1 | tee -a pipeline.log
echo "[chain] Stage 5 done."

echo "[chain] Running Stage 6..."
python3 run_pipeline.py --stages 6 2>&1 | tee -a pipeline.log
echo "[chain] Stage 6 done."

echo "=============================="
echo " FULL PIPELINE COMPLETE"
echo "=============================="
python3 -c "
import json, os
outputs = {
    'Stage 2': ('data/stage2_output/distilled_logic.json', lambda d: f'{sum(1 for v in d.values() if \"Parameter\" in v or \"optimize\" in v)} real distillations'),
    'Stage 3': ('data/stage3_output/top50_pairs.json', lambda d: f'{len(d)} cross-domain pairs'),
    'Stage 4': ('data/stage4_output/verified_pairs.json', lambda d: f'{len(d)} verified pairs'),
    'Stage 5': ('data/stage5_output/missing_links.json', lambda d: f'{len(d)} predictions'),
}
for stage, (path, fn) in outputs.items():
    if os.path.exists(path):
        with open(path) as f: d = json.load(f)
        print(f'  {stage}: {fn(d)}')
if os.path.exists('data/stage6_output/hypotheses.md'):
    size = os.path.getsize('data/stage6_output/hypotheses.md')
    print(f'  Stage 6: hypotheses.md ({size} bytes)')
"
