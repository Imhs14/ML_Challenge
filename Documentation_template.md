# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Emberline
**Team Members:** Rajneesh Kaur, Pravesh Kumar, Malathkar Heera Shanker, Ikshit Sinha.
**Submission Date:** 27-Sep-2026

---

## 1. Executive Summary

Our solution frames entity resolution as a two-stage **blocking + classifier** pipeline. Candidate generation uses a per-country TF-IDF nearest-neighbour index to cut the comparison space from billions of possible pairs down to ~34.6M candidates on the test set, and a LightGBM binary classifier scores each candidate pair using a 9-dimensional similarity feature vector. The match/no-match threshold was tuned directly against the macro F_0.5 objective on a held-out validation split, converging on **threshold = 0.85** for a **macro F_0.5 of 0.7682**. Because blocking and feature generation are keyed on the `country` field generically rather than hard-coded to `{US, India}`, the pipeline transferred without modification to the unseen `France` test country.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA surfaced two dominant noise categories, which is reflected directly in the normalization dictionaries built into `features.py`:

- **Business-name noise:** legal-form and organizational abbreviations — `Corp`, `Inc`, `Ltd`, `LLC`, `LLP`, `Pvt`, `Co`, `Bros`, `Intl`, `Mfg`, `Svc(s)`, `Grp`, `Asso(c)` — plus `DBA` markers and leading articles (`The`), and the `&`/"and" punctuation split. All of these are canonicalized to a single expanded form before comparison (e.g. `Pvt` → `private`, `&` → `and`) rather than left for the similarity metrics to reconcile.
- **Address noise:** street/unit-type abbreviations (`Rd`, `St`, `Ave`, `Blvd`, `Dr`, `Ln`, `Ct`, `Pl`, `Sq`, `Pkwy`, `Fwy`, `Hwy`, `Expy`, `Ste`, `Apt`, `Fl`) and directional abbreviations (`N/S/E/W`, `NE/NW/SE/SW`), which are expanded the same way.
- Both fields also go through Unicode `NFKC` normalization (to fold accented/transliterated characters) and punctuation stripping before any similarity metric is computed, addressing the transliteration and formatting-variation noise called out in the problem statement.

In the training data, these patterns are common enough that leaving them unnormalized would materially hurt recall. Across the combined name/address text for Source 1, legal-form abbreviations were frequent: `llc` appeared in 17.1% of records, `inc` in 12.0%, `ltd` in 7.1%, `pvt` in 5.6%, and `corp` in 2.7%. Address abbreviations were equally pervasive: `st` appeared in 47.4% of Source-1 records, `rd` in 11.4%, and `avenue`/`ave` in 8.7% of rows. The motivation for the normalization rules is visible in concrete examples such as `Custom Wealth Services LLC` vs `Custom Wealth Services`, `B+ Retail Inc` vs `B+ Retail`, and `11643 Prosperity Road` vs `11643 Prosperity Rd`; without canonicalization, those would be treated as distinct names and streets even though they describe the same underlying business. Likewise, directional and unit variants such as `APT 5`, `Fl A-105`, `Unit APT 5`, or `Near SBI ATM` are common in the data and were folded into a single normalized form before scoring.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier

**Core Innovation:** A country-partitioned TF-IDF blocking stage (a separate index built for each `country` value, queried independently against Source 2 and Source 3) combined with a LightGBM matcher whose decision threshold is tuned directly against the competition's macro F_0.5 metric rather than a generic 0.5 cutoff. Because the blocking/indexing code treats `country` as an arbitrary string key instead of a fixed `{US, India}` category, it required no changes to generalize to the previously-unseen `France` records at test time — the run log shows `Countries in S1: {'us', 'india', 'france'}` being picked up automatically during test-set processing.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Country partition (`country` field, case-insensitive) + character-level TF-IDF cosine similarity over a **combined name + normalized-address text field** (`combined_text = normalize_name(name) + " " + normalize_address(address)`). A separate `TfidfVectorizer` index is fit per country, per source (one against Source 2, one against Source 3), using: `analyzer='char_wb'`, `ngram_range=(3, 5)`, `min_df=2`, `max_df=0.3`, `max_features=100,000`, `sublinear_tf=True`. Character n-grams (rather than word tokens) were chosen specifically to stay robust to typos, transliteration, and abbreviation noise that word-level TF-IDF would be more brittle to.
  - **Top-k cutoff:** the **top 10** highest-cosine Sx records are retrieved per S1 entity, per source (so up to 20 raw candidates per S1 entity across S2+S3 before de-duplication), with zero-score matches discarded.
  - **Query-side pruning:** before the sparse dot product, each S1 query vector is pruned to its top 15 highest-TF-IDF terms — a speed optimization for the sparse matrix multiply at this scale, applied in batches of 5,000 queries.
- **Candidate pairs generated:**
  - Train: 27,096,185 pairs (pos = 5,059,069 / neg = 22,037,116)
  - Validation: 3,788,887 pairs (pos = 559,695 / neg = 3,229,192)
  - Test: 34,650,880 pairs, generated for all 1,732,544 test Source-1 entities
- **How you ensured true matches were not lost:** Ground-truth positives are joined against the blocking output (`build_train_pairs`) — any ground-truth match not retrieved by the TF-IDF top-10 blocking step is simply absent from the candidate set and cannot be recovered downstream, so blocking recall is the hard ceiling on the whole pipeline. The blocking stage recovered 5,059,069 positive pairs on the train split and 559,695 on validation. The blocking ceiling is explicit once we divide the recovered positives by the total ground-truth positive count in the training set. There are 7,638,365 true matches in `train_ground_truth.tsv` overall, and the blocking stage recovered 5,059,069 of them, so the candidate generation step achieved a recall ceiling of 5,059,069 / 7,638,365 = 0.6623, or about 66.23% of all true matches before the classifier even runs. On the validation split, the same calculation gives 559,695 recovered positives out of 762,747 ground-truth positives, i.e. a blocking recall ceiling of 73.38%.

---

## 4. Matching Model

**Features used:** Exactly 9 features (`FEATURE_NAMES` in `features.py`), computed per candidate pair on the abbreviation-normalized name/address strings:
- Name features: `name_token_jaccard` (word-set Jaccard), `name_char3_jaccard` (character 3-gram Jaccard), `name_seq_ratio` (`difflib.SequenceMatcher` ratio), `name_token_sort_ratio` (sequence ratio after sorting tokens alphabetically, to neutralize word-order transpositions)
- Address features: `addr_token_jaccard`, `addr_char3_jaccard`, `addr_seq_ratio` — the same three metric families applied to the normalized address string (no token-sort variant for address)
- Other: `country_match` (1.0/0.0 exact match on the lower-cased `country` field), `tfidf_cosine` (the blocking-stage cosine similarity score, passed through as a 9th feature so the classifier can directly weigh blocking confidence)

**Model type:** LightGBM (`LGBMClassifier`) binary classifier: `n_estimators=500`, `learning_rate=0.05`, `num_leaves=63`, `max_depth=-1` (unlimited), `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`. Class imbalance (candidate sets are dominated by non-matches) is handled via `scale_pos_weight = n_negative / n_positive` rather than resampling. Training pairs themselves are also pre-balanced at the sampling stage: for each S1 entity, negatives are capped at `neg_ratio × (positives found in its candidate set)` — **5×** for the training split and **10×** for the validation split — rather than using every negative candidate.

**Threshold selection method:** Grid search over classifier probability thresholds from 0.10 to 0.90 (step 0.05, `np.arange(0.1, 0.95, 0.05)`), evaluated as macro-averaged F_0.5 (computed per S1 entity via the exact competition formula, including the singleton edge case) on a held-out validation split (10% of training S1 entities, `random_state=42`). Threshold = 0.85 gave the best score before performance began to fall (0.90 scored lower), and was locked in for test-set inference.

| Threshold | Macro F_0.5 (val) |
|---|---|
| 0.10 | 0.6299 |
| 0.20 | 0.6688 |
| 0.30 | 0.6912 |
| 0.40 | 0.7083 |
| 0.50 | 0.7233 |
| 0.60 | 0.7369 |
| 0.70 | 0.7502 |
| 0.80 | 0.7629 |
| **0.85** | **0.7682 (best)** |
| 0.90 | 0.7670 |

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.7682, on the held-out validation split (220,682 Source-1 entities), at threshold 0.85.
- **Common false positives (wrong merges):** The validation error analysis points to branch/franchise businesses as the dominant source of over-merging. In the same-country setting, many chain businesses share highly similar names and even near-identical landmark-based address strings, so the model can assign a high probability to a wrong-branch match when `name_token_jaccard`, `name_char3_jaccard`, and `tfidf_cosine` all remain high but the address signal is not discriminative enough. In practice, this manifests as cross-location matches of the same brand, especially when both records live in the same city or share a recognizable landmark.
- **Common false negatives (missed matches):** The main failure mode is still upstream of the classifier: (1) the fixed top-10-per-source blocking cutoff drops a true match if it falls just outside the top 10 by TF-IDF cosine, and (2) the per-query pruning down to the top 15 weighted character terms can remove the discriminative signal for longer names or address strings where the distinguishing tokens are not among the top terms. Ground-truth positives falling below 0.85 in the classifier are also an issue, but the strongest structural cause is that a large portion of missed matches never reach the classifier at all because of the blocking ceiling itself.

The threshold sweep itself is informative: F_0.5 rises steadily and smoothly from 0.63 to 0.7682 as the threshold increases from 0.10 to 0.85, consistent with F_0.5's precision-heavy weighting rewarding a stricter cutoff — until 0.90, where it dips slightly, indicating the classifier starts sacrificing recall (dropping true matches, which is heavily penalized for singleton-heavy evaluation) faster than it gains precision beyond 0.85.

---

## 6. Conclusion

The pipeline pairs a scalable, country-agnostic TF-IDF blocking stage with a LightGBM classifier whose decision threshold is tuned directly against the competition metric, reaching a macro F_0.5 of 0.7682 on validation. Its main strength is generalizing to an unseen country (France) at test time without pipeline changes, since country is treated as an open string key throughout. The main lesson learned was around compute cost: end-to-end blocking and scoring across ~34.6M test candidate pairs took roughly 12 hours (723.7 minutes total), driven by exact sparse cosine similarity search over millions of rows per country.

With more time, we would prioritize: (1) replacing the exact TF-IDF sparse-matrix blocking with an approximate-nearest-neighbour index (e.g. FAISS or ScaNN) to cut blocking time from hours to minutes and afford a higher top-k without a runtime penalty, since the current fixed top-10-per-source cutoff is a likely source of missed matches (Section 5); (2) raising `TOP_K` and/or relaxing the top-15-term query pruning to test whether blocking recall — not just classifier precision — is the binding constraint on the current F_0.5; (3) adding features the current 9-feature set lacks entirely, such as phonetic name encoding (Soundex/Metaphone/Double Metaphone) for typo-heavy transliterations, and an exact/fuzzy postal-code or PIN-code match feature, since address similarity today is purely lexical (Jaccard/char n-gram/sequence-ratio) with no explicit code-level signal; and (4) tuning the classification threshold per-country rather than globally, given that name/address noise characteristics differ by country and France was never seen during threshold tuning.

---

## Appendix

### A. Code Artefacts

Code lives under `code/business_entity_resolution/src/`, split into two modules:
- **`features.py`** — text normalization (abbreviation expansion, Unicode/punctuation cleanup) and the 9-feature pairwise similarity computation (`compute_features`, `FEATURE_NAMES`).
- **`pipeline.py`** — the end-to-end 8-stage orchestration: (1) load & normalize training data, (2) load ground truth, (3) train/val split (90/10, seed 42), (4) per-country TF-IDF blocking to generate candidates, (5) build the training/validation feature matrices, (6) train LightGBM, (7) tune the decision threshold on validation, (8) process test data (normalize → block → score → write `matching_results.tsv` and `candidate_pairs.tsv`).

**Entry point** (run from the `student_resource/` root, per the script's own usage docstring):
```bash
./venv/bin/python code/business_entity_resolution/src/pipeline.py
```
Intermediate artifacts (normalized data, candidate sets, feature matrices, the trained model) are pickle-cached under a local `.cache/` directory so re-runs skip already-completed stages.

**Dependencies** used directly by the pipeline: `numpy`, `scipy` (`scipy.sparse`), `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`), and `lightgbm`. The exact runtime versions in the project environment were: `numpy==2.4.6`, `scipy==1.17.1`, `scikit-learn==1.9.1`, and `lightgbm==4.7.0`. LightGBM is distributed under the MIT license, and the model is a gradient-boosted tree ensemble rather than a large transformer or dense neural net, so it does not have a meaningful "parameter count" in the LLM sense and is compliant with the challenge's model-size constraint. The project’s checked-in `requirements.txt` uses semver ranges, but the operationally validated lock for this run was the environment above.

### B. Additional Results

Dataset scale processed by this run: test Source 1 = 1,732,544 rows, Source 2 = 4,887,273 rows, Source 3 = 5,082,316 rows. Training candidate generation covered `{us, india}`; test candidate generation additionally covered `{france}` with no code changes. The threshold sweep in Section 4 is the main diagnostic plot for the model: F_0.5 climbs from 0.6299 at 0.10 to 0.7682 at 0.85, then drops slightly to 0.7670 at 0.90, indicating that precision is being rewarded while the recall penalty begins to dominate beyond the optimal threshold. This is consistent with the competition metric’s emphasis on precision: the probability cutoff is tuned to maximize macro F_0.5, not to maximize raw recall or a generic 0.5 decision boundary.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
