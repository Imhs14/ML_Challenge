# Business Entity Resolution — Code

## How to Reproduce

### 1. Set up environment

```bash
cd ML_Challenge_student_resource/
python3 -m venv venv
venv/bin/pip install -r code/business_entity_resolution/requirements.txt
```

On macOS, LightGBM requires `libomp`:
```bash
brew install libomp
```

### 2. Run the pipeline

```bash
venv/bin/python code/business_entity_resolution/src/pipeline.py
```

This produces:
- `output/matching_results.tsv` — leaderboard submission file
- `output/candidate_pairs.tsv` — blocking audit file

### 3. Validate the output

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## Pipeline Overview

```
[1] Load & normalize (abbreviation expansion, unicode normalization)
[2] Country-partitioned TF-IDF blocking (char 2-4 grams, top-10 per S1)
[3] Pairwise feature engineering (token Jaccard, char Jaccard, seq ratio, TF-IDF cosine)
[4] LightGBM binary classifier (500 trees, balanced class weight)
[5] F₀.₅-tuned decision threshold on 10% validation split
[6] Test prediction + output generation
```

## File Structure

```
src/
  pipeline.py   — main orchestrator (all 8 stages)
  features.py   — normalization + pairwise feature computation
requirements.txt
README.md
```
