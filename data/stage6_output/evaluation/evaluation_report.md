# Hypothesis Evaluation Report

**Method:** Each hypothesis was scored by `qwen3.5:2b` (local Ollama) on 5 scientific dimensions (1–5 scale).  
**Radar charts:** Individual and combined charts saved in `evaluation/figures/`.

## Summary Scores

| # | Paper A | Paper B | Novelty | Significance | Effectiveness | Clarity | Feasibility | **Average** |
|---|---------|---------|---------|--------------|---------------|---------|-------------|-------------|
| 1 | `2789744648` | `2950181023` | 4.0 | 5.0 | 4.0 | 5.0 | 3.0 | **4.2** |
| 2 | `2950628590` | `2917805034` | 4.0 | 4.0 | 3.0 | 5.0 | 2.0 | **3.6** |
| 3 | `2789744648` | `2917805034` | 4.0 | 5.0 | 3.0 | 5.0 | 4.0 | **4.2** |
| 4 | `3003152535` | `2950129161` | 5.0 | 5.0 | 4.0 | 5.0 | 3.0 | **4.4** |
| 5 | `2951659998` | `2789744648` | 4.0 | 5.0 | 4.0 | 5.0 | 3.0 | **4.2** |

## Per-Hypothesis Scores

### Hypothesis 1
- **Papers:** `2789744648` (cs.GR) ↔ `2950181023` (cs.MA)
- **Embedding similarity:** 0.9541
- **Structural overlap:** 0.1667
- **Novelty:** 4.0/5  
- **Significance:** 5.0/5  
- **Effectiveness:** 4.0/5  
- **Clarity:** 5.0/5  
- **Feasibility:** 3.0/5  
- **Average:** **4.2/5**
- **Radar chart:** `evaluation/hypothesis_01_radar.png`

### Hypothesis 2
- **Papers:** `2950628590` (cs.GL) ↔ `2917805034` (cs.MA)
- **Embedding similarity:** 0.9397
- **Structural overlap:** 0.1333
- **Novelty:** 4.0/5  
- **Significance:** 4.0/5  
- **Effectiveness:** 3.0/5  
- **Clarity:** 5.0/5  
- **Feasibility:** 2.0/5  
- **Average:** **3.6/5**
- **Radar chart:** `evaluation/hypothesis_02_radar.png`

### Hypothesis 3
- **Papers:** `2789744648` (cs.GR) ↔ `2917805034` (cs.MA)
- **Embedding similarity:** 0.9513
- **Structural overlap:** 0.0952
- **Novelty:** 4.0/5  
- **Significance:** 5.0/5  
- **Effectiveness:** 3.0/5  
- **Clarity:** 5.0/5  
- **Feasibility:** 4.0/5  
- **Average:** **4.2/5**
- **Radar chart:** `evaluation/hypothesis_03_radar.png`

### Hypothesis 4
- **Papers:** `3003152535` (cs.NA) ↔ `2950129161` (cs.GL)
- **Embedding similarity:** 0.9537
- **Structural overlap:** 0.0833
- **Novelty:** 5.0/5  
- **Significance:** 5.0/5  
- **Effectiveness:** 4.0/5  
- **Clarity:** 5.0/5  
- **Feasibility:** 3.0/5  
- **Average:** **4.4/5**
- **Radar chart:** `evaluation/hypothesis_04_radar.png`

### Hypothesis 5
- **Papers:** `2951659998` (cs.GL) ↔ `2789744648` (cs.GR)
- **Embedding similarity:** 0.9651
- **Structural overlap:** 0.0769
- **Novelty:** 4.0/5  
- **Significance:** 5.0/5  
- **Effectiveness:** 4.0/5  
- **Clarity:** 5.0/5  
- **Feasibility:** 3.0/5  
- **Average:** **4.2/5**
- **Radar chart:** `evaluation/hypothesis_05_radar.png`

## Combined Radar Chart

![Combined Radar](evaluation/all_hypotheses_radar.png)

_All hypotheses plotted on a single radar for direct comparison._
