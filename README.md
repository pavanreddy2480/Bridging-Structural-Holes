# Analogical Link Prediction Pipeline

**Discovering Inter-Domain Structural Holes via Stratified LLM Distillation and Analogical Link Prediction**

Version 5.0 — 21 patches applied.

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Configure API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and S2_API_KEY

# 4. Run the full pipeline
python3 run_pipeline.py

# 5. Results
cat data/stage6_output/hypotheses.md
```

## Resuming from a Specific Stage

```bash
python3 run_pipeline.py --start-stage 4    # Resume from Stage 4
python3 run_pipeline.py --stages 5 6       # Re-run only Stage 5 and 6
```

## Pipeline Overview

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `src/stage1_tfidf_filter.py` | TF-IDF (SnowballStemmer) → top 2,000 method-dense papers |
| 2 | `src/stage2_llm_distillation.py` | Async GPT-4o-mini distillation → domain-blind logic strings |
| 3 | `src/stage3_pair_extraction.py` | MiniLM embeddings → cross-domain pairs + citation chasm filter |
| 4 | `src/stage4_pdf_encoding.py` | PDF download → spaCy dependency trees → Jaccard verification |
| 5 | `src/stage5_link_prediction.py` | Bidirectional graph analysis → structural hole predictions |
| 6 | `src/stage6_hypothesis_synthesis.py` | GPT-4o synthesis → publishable research hypotheses |

See `CORRECTED_IMPLEMENTATION_PLAN_v4.md` for full technical details.
