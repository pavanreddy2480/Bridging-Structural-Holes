# Pipeline Output Package
Generated: 2026-04-10 00:16:34

## Contents
| File | Description |
|------|-------------|
| `hypotheses.md` | 5 LLM-generated cross-domain research hypotheses |
| `evaluation_report.md` | Per-hypothesis scores (Novelty/Significance/Effectiveness/Clarity/Feasibility) |
| `figures/all_hypotheses_radar.png` | Combined radar chart for all 5 hypotheses |
| `figures/hypothesis_NN_radar.png` | Individual radar chart per hypothesis |
| `data/all_results.json` | Consolidated machine-readable results from all stages |
| `pipeline.log` | Full execution log with DEBUG detail |

## Pipeline Summary
- **Stage 1:** 2,000 papers filtered from OGBN-ArXiv (169,343 total)
- **Stage 2:** Domain-neutral logic distillation via Ollama qwen3.5:2b
- **Stage 3:** Cross-domain pair extraction (cosine similarity + citation chasm filter)
- **Stage 4:** Methodology PDF extraction + spaCy dependency parsing
- **Stage 5:** Bidirectional analogical missing-link prediction
- **Stage 6:** 5 structured research hypotheses generated
- **Stage 7:** Multi-dimension evaluation + radar chart visualisation
