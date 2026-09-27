#!/usr/bin/env python3
"""
pipeline.py — End-to-end Business Entity Resolution pipeline.

Usage (from the student_resource/ root):
    ./venv/bin/python code/business_entity_resolution/src/pipeline.py

Outputs:
    output/matching_results.tsv
    output/candidate_pairs.tsv
"""
from __future__ import annotations

import csv
import gc
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from features import (
    FEATURE_NAMES,
    combined_text,
    compute_features,
    normalize_address,
    normalize_name,
)

import lightgbm as lgb

# ── config ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]  # student_resource/
TRAIN_DIR = ROOT / "dataset" / "train"
TEST_DIR  = ROOT / "dataset" / "test"
OUT_DIR   = ROOT / "output"
CACHE_DIR = ROOT / ".cache"

OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

TOP_K          = 10    # candidates per S1 entity from each of S2, S3
BATCH_SIZE     = 2000  # S1 rows to query per cosine-sim batch
VAL_FRAC       = 0.10  # fraction of training data held out for threshold tuning
RANDOM_SEED    = 42
NEG_RATIO      = 5     # negative samples per positive in training

# ── helpers ──────────────────────────────────────────────────────────────────

def read_tsv(path: Path) -> List[dict]:
    """Read a TSV file into a list of dicts."""
    t0 = time.time()
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(row)
    log.info(f"  read {len(rows):,} rows from {path.name}  ({time.time()-t0:.1f}s)")
    return rows


def build_lookup(rows: List[dict]) -> Dict[str, dict]:
    return {r['entity_id']: r for r in rows}


def cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.pkl"


def save_cache(obj, name: str):
    with open(cache_path(name), 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    log.info(f"  cached {name}")


def load_cache(name: str):
    p = cache_path(name)
    if p.exists():
        with open(p, 'rb') as f:
            return pickle.load(f)
    return None


# ── stage 1: load & normalize ────────────────────────────────────────────────

def load_and_normalize(split: str) -> Tuple[dict, dict, dict]:
    """
    Returns (s1_lookup, s2_lookup, s3_lookup) where each value is a dict:
        entity_id → {entity_id, business_name, business_address, country,
                      name_norm, addr_norm, combined}
    """
    cache_key = f"normalized_{split}"
    cached = load_cache(cache_key)
    if cached:
        log.info(f"  loaded normalized {split} from cache")
        return cached

    log.info(f"Loading & normalizing {split} data …")
    if split == "train":
        s1_raw = read_tsv(TRAIN_DIR / "train_source1.tsv")
        s2_raw = read_tsv(TRAIN_DIR / "train_source2.tsv")
        s3_raw = read_tsv(TRAIN_DIR / "train_source3.tsv")
    else:
        s1_raw = read_tsv(TEST_DIR / "test_source1.tsv")
        s2_raw = read_tsv(TEST_DIR / "test_source2.tsv")
        s3_raw = read_tsv(TEST_DIR / "test_source3.tsv")

    def enrich(rows):
        lookup = {}
        for r in rows:
            r['name_norm'] = normalize_name(r.get('business_name', '') or '')
            r['addr_norm'] = normalize_address(r.get('business_address', '') or '')
            r['combined']  = combined_text(r.get('business_name', '') or '',
                                           r.get('business_address', '') or '')
            r['country']   = (r.get('country') or '').strip()
            lookup[r['entity_id']] = r
        return lookup

    log.info("  Enriching S1 …")
    s1 = enrich(s1_raw)
    log.info("  Enriching S2 …")
    s2 = enrich(s2_raw)
    log.info("  Enriching S3 …")
    s3 = enrich(s3_raw)

    result = (s1, s2, s3)
    save_cache(result, cache_key)
    return result


# ── stage 2: blocking (TF-IDF cosine) ────────────────────────────────────────

def build_tfidf_index(records: List[dict], country: str) -> Tuple[TfidfVectorizer, csr_matrix, List[str]]:
    """
    Fit TF-IDF on combined texts for a given country partition.
    Returns (vectorizer, matrix, id_list).
    """
    subset = [r for r in records if r['country'].lower() == country.lower()]
    if not subset:
        return None, None, []
    texts  = [r['combined'] for r in subset]
    ids    = [r['entity_id'] for r in subset]
    vect   = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.3,
        max_features=100_000,
        sublinear_tf=True,
    )
    mat = vect.fit_transform(texts)
    return vect, mat, ids


def retrieve_candidates(
    s1_rows: List[dict],
    sx_rows: List[dict],
    country: str,
    top_k: int = TOP_K,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    For each S1 entity in `country`, retrieve top_k candidates from sx_rows.
    Returns {s1_id: [(sx_id, cosine_score), …]}.
    """
    log.info(f"    Building Sx TF-IDF index for country='{country}' …")
    vect, sx_mat, sx_ids = build_tfidf_index(sx_rows, country)
    if vect is None:
        return {}

    s1_subset = [r for r in s1_rows if r['country'].lower() == country.lower()]
    if not s1_subset:
        return {}

    log.info(f"    Querying {len(s1_subset):,} S1 rows against {len(sx_ids):,} Sx rows …")
    results: Dict[str, List[Tuple[str, float]]] = {}

    s1_texts = [r['combined'] for r in s1_subset]
    s1_ids_  = [r['entity_id'] for r in s1_subset]
    sx_ids_arr = np.array(sx_ids)

    # Convert sx_mat transpose once to CSR for fast sparse dot product
    sx_mat_T = sx_mat.T.tocsr()

    SPARSE_BATCH_SIZE = 5000

    # Process in batches to avoid OOM
    for start in range(0, len(s1_subset), SPARSE_BATCH_SIZE):
        end   = min(start + SPARSE_BATCH_SIZE, len(s1_subset))
        batch_texts = s1_texts[start:end]
        batch_ids   = s1_ids_[start:end]

        q_mat = vect.transform(batch_texts)

        # Prune q_mat: keep top 15 highest TF-IDF terms per query row
        q_coo = q_mat.tocoo()
        row_terms = defaultdict(list)
        for r, c, v in zip(q_coo.row, q_coo.col, q_coo.data):
            row_terms[r].append((v, c))

        r_p, c_p, v_p = [], [], []
        for r, terms in row_terms.items():
            if len(terms) > 15:
                terms.sort(key=lambda x: x[0], reverse=True)
                terms = terms[:15]
            for v, c in terms:
                r_p.append(r)
                c_p.append(c)
                v_p.append(v)

        q_pruned = csr_matrix((v_p, (r_p, c_p)), shape=q_mat.shape)

        scores_csr = q_pruned.dot(sx_mat_T)  # sparse dot product with pruned terms

        indptr = scores_csr.indptr
        indices = scores_csr.indices
        data = scores_csr.data

        for i, s1_id in enumerate(batch_ids):
            row_start = indptr[i]
            row_end = indptr[i+1]
            if row_start == row_end:
                results[s1_id] = []
                continue

            row_cols = indices[row_start:row_end]
            row_vals = data[row_start:row_end]

            n_elements = len(row_vals)
            if n_elements <= top_k:
                top_idx = np.argsort(row_vals)[::-1]
            else:
                top_idx = np.argpartition(row_vals, -top_k)[-top_k:]
                top_idx = top_idx[np.argsort(row_vals[top_idx])[::-1]]

            candidates = [
                (sx_ids_arr[row_cols[j]], float(row_vals[j]))
                for j in top_idx
                if row_vals[j] > 0.0
            ]
            results[s1_id] = candidates

        if (start // SPARSE_BATCH_SIZE) % 5 == 0 or end == len(s1_subset):
            log.info(f"      … {end:,}/{len(s1_subset):,}")

    return results


def generate_candidates(
    s1: dict, s2: dict, s3: dict, split: str
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Full blocking stage. Returns {s1_id: [(sx_id, score), …]} merged from S2+S3.
    """
    cache_key = f"candidates_{split}"
    cached = load_cache(cache_key)
    if cached:
        log.info(f"  loaded candidates_{split} from cache")
        return cached

    log.info(f"Generating candidates for {split} …")

    s1_rows = list(s1.values())
    s2_rows = list(s2.values())
    s3_rows = list(s3.values())

    # Discover all countries present
    countries = set(r['country'].lower() for r in s1_rows)
    log.info(f"  Countries in S1: {countries}")

    all_candidates: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    for country in countries:
        log.info(f"  === Country: {country} ===")
        log.info(f"  Blocking against S2 …")
        c2 = retrieve_candidates(s1_rows, s2_rows, country, TOP_K)
        log.info(f"  Blocking against S3 …")
        c3 = retrieve_candidates(s1_rows, s3_rows, country, TOP_K)

        # Merge S2 and S3 candidates per S1 entity
        for s1_id, cands in c2.items():
            all_candidates[s1_id].extend(cands)
        for s1_id, cands in c3.items():
            all_candidates[s1_id].extend(cands)

    # Fill in S1 entities with no candidates (singletons / unseen country in Sx)
    for s1_id in s1:
        if s1_id not in all_candidates:
            all_candidates[s1_id] = []

    result = dict(all_candidates)
    save_cache(result, cache_key)
    log.info(f"  Generated candidates for {len(result):,} S1 entities")
    return result


# ── stage 3: feature engineering ─────────────────────────────────────────────

def build_feature_matrix(
    pairs: List[Tuple[str, str, float, int]],  # (s1_id, sx_id, cosine, label)
    s1: dict,
    sx_all: dict,  # merged S2+S3 lookup
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Compute feature matrix for a list of (s1_id, sx_id, cosine, label) pairs.
    Returns (X, y, pair_ids).
    """
    X, y, pair_ids = [], [], []
    for s1_id, sx_id, cosine, label in pairs:
        s1r = s1.get(s1_id)
        sxr = sx_all.get(sx_id)
        if s1r is None or sxr is None:
            continue
        feats = compute_features(
            s1r['name_norm'], s1r['addr_norm'], s1r['country'],
            sxr['name_norm'], sxr['addr_norm'], sxr['country'],
            tfidf_cosine=cosine,
        )
        X.append(feats)
        y.append(label)
        pair_ids.append((s1_id, sx_id))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int8), pair_ids


# ── stage 4: build training pairs ────────────────────────────────────────────

def load_ground_truth(path: Path) -> Dict[str, List[str]]:
    gt = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            matched = row['matched_entity_ids'].strip()
            gt[row['source1_entity_id']] = matched.split(',') if matched else []
    return gt


def build_train_pairs(
    s1: dict,
    candidates: Dict[str, List[Tuple[str, float]]],
    gt: Dict[str, List[str]],
    neg_ratio: int = NEG_RATIO,
) -> List[Tuple[str, str, float, int]]:
    """
    Build (s1_id, sx_id, cosine, label) pairs for training.
    Positives = ground truth matches that appear in candidates.
    Negatives = sampled non-matching candidates.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    pairs = []

    for s1_id, cands in candidates.items():
        pos_set = set(gt.get(s1_id, []))
        cand_dict = {sx_id: score for sx_id, score in cands}

        # Positives: ground-truth matches found in candidates
        for sx_id, score in cand_dict.items():
            if sx_id in pos_set:
                pairs.append((s1_id, sx_id, score, 1))

        # Negatives: candidates that are not true matches
        neg_cands = [(sx_id, score) for sx_id, score in cand_dict.items()
                     if sx_id not in pos_set]
        n_neg = min(len(neg_cands), neg_ratio * max(1, len(pos_set & set(cand_dict))))
        if n_neg > 0:
            chosen = rng.choice(len(neg_cands), size=n_neg, replace=False)
            for i in chosen:
                sx_id, score = neg_cands[i]
                pairs.append((s1_id, sx_id, score, 0))

    log.info(f"  Training pairs: {len(pairs):,}  "
             f"(pos={sum(p[3] for p in pairs):,}, neg={sum(1-p[3] for p in pairs):,})")
    return pairs


# ── stage 5: train LightGBM ───────────────────────────────────────────────────

def train_model(X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    pos = y.sum()
    neg = len(y) - pos
    scale = neg / max(pos, 1)

    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X, y)
    log.info("  LightGBM trained")
    return clf


# ── stage 6: F₀.₅ threshold tuning ───────────────────────────────────────────

def f05_score_entity(pred_set: set, true_set: set) -> float:
    if not true_set and not pred_set:
        return 1.0
    if not true_set and pred_set:
        return 0.0
    if not pred_set:
        tp, fp, fn = 0, 0, len(true_set)
    else:
        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return (1.25 * prec * rec) / (0.25 * prec + rec)


def tune_threshold(
    s1_ids: List[str],
    pair_ids: List[Tuple[str, str]],
    proba: np.ndarray,
    gt: Dict[str, List[str]],
    thresholds=None,
) -> float:
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    best_thresh, best_f = 0.5, 0.0

    for thresh in thresholds:
        # Build per-S1 predictions
        preds: Dict[str, set] = {s1_id: set() for s1_id in s1_ids}
        for (s1_id, sx_id), p in zip(pair_ids, proba):
            if s1_id in preds and p >= thresh:
                preds[s1_id].add(sx_id)

        scores = []
        for s1_id in s1_ids:
            true_set = set(gt.get(s1_id, []))
            pred_set = preds.get(s1_id, set())
            scores.append(f05_score_entity(pred_set, true_set))

        macro_f = np.mean(scores)
        if macro_f > best_f:
            best_f     = macro_f
            best_thresh = thresh
        log.info(f"    threshold={thresh:.2f}  macro_F0.5={macro_f:.4f}")

    log.info(f"  Best threshold: {best_thresh:.2f}  (macro F0.5={best_f:.4f})")
    return float(best_thresh)


# ── stage 7: predict on test ──────────────────────────────────────────────────

def predict(
    s1: dict,
    sx_all: dict,
    candidates: Dict[str, List[Tuple[str, float]]],
    model: lgb.LGBMClassifier,
    threshold: float,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Returns (matching_results, candidate_pairs) both as {s1_id: [sx_ids]}.
    """
    log.info(f"  Predicting with threshold={threshold:.2f} …")

    matching: Dict[str, List[str]]   = {}
    cand_out: Dict[str, List[str]]   = {}

    INFER_BATCH = 50_000
    all_pairs: List[Tuple[str, str, float]] = []
    for s1_id, cands in candidates.items():
        for sx_id, score in cands:
            all_pairs.append((s1_id, sx_id, score))

    log.info(f"  Total candidate pairs to score: {len(all_pairs):,}")

    # Score in batches
    scores_out: Dict[Tuple[str, str], float] = {}
    for start in range(0, len(all_pairs), INFER_BATCH):
        batch = all_pairs[start:start + INFER_BATCH]
        raw_pairs = [(s1_id, sx_id, score, 0) for s1_id, sx_id, score in batch]
        X, _, pair_ids = build_feature_matrix(raw_pairs, s1, sx_all)
        if len(X) == 0:
            continue
        proba = model.predict_proba(X)[:, 1]
        for (s1_id, sx_id), p in zip(pair_ids, proba):
            scores_out[(s1_id, sx_id)] = float(p)

        if start % (INFER_BATCH * 10) == 0:
            log.info(f"    … scored {start + len(batch):,}/{len(all_pairs):,}")

    # Apply threshold
    for s1_id, cands in candidates.items():
        cand_list = [sx_id for sx_id, _ in cands]
        match_list = [
            sx_id for sx_id in cand_list
            if scores_out.get((s1_id, sx_id), 0.0) >= threshold
        ]
        # Deduplicate, preserve order
        seen = set()
        unique_cands = []
        for sx_id in cand_list:
            if sx_id not in seen:
                seen.add(sx_id)
                unique_cands.append(sx_id)
        seen = set()
        unique_matches = []
        for sx_id in match_list:
            if sx_id not in seen:
                seen.add(sx_id)
                unique_matches.append(sx_id)

        cand_out[s1_id]  = unique_cands
        matching[s1_id]  = unique_matches

    return matching, cand_out


# ── stage 8: write outputs ────────────────────────────────────────────────────

def write_outputs(
    matching: Dict[str, List[str]],
    candidates: Dict[str, List[str]],
    s1_all_ids: List[str],
):
    mr_path   = OUT_DIR / "matching_results.tsv"
    cand_path = OUT_DIR / "candidate_pairs.tsv"

    with open(mr_path, 'w', newline='', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_all_ids:
            ids = matching.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(ids)}\n")

    with open(cand_path, 'w', newline='', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_all_ids:
            ids = candidates.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(ids)}\n")

    log.info(f"  Wrote {mr_path}")
    log.info(f"  Wrote {cand_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    log.info("=" * 60)
    log.info("Business Entity Resolution Pipeline")
    log.info("=" * 60)

    # ── 1. Load & normalize training data ──────────────────────────
    log.info("\n[1/8] Loading & normalizing training data …")
    s1_tr, s2_tr, s3_tr = load_and_normalize("train")
    sx_tr = {**s2_tr, **s3_tr}   # merged S2+S3 lookup

    # ── 2. Load ground truth ───────────────────────────────────────
    log.info("\n[2/8] Loading ground truth …")
    gt = load_ground_truth(TRAIN_DIR / "train_ground_truth.tsv")

    # ── 3. Train/val split ─────────────────────────────────────────
    log.info("\n[3/8] Train/val split …")
    rng = np.random.default_rng(RANDOM_SEED)
    all_s1_train_ids = list(s1_tr.keys())
    rng.shuffle(all_s1_train_ids)
    n_val = int(len(all_s1_train_ids) * VAL_FRAC)
    val_ids  = set(all_s1_train_ids[:n_val])
    train_ids = set(all_s1_train_ids[n_val:])
    log.info(f"  train={len(train_ids):,}  val={len(val_ids):,}")

    s1_tr_split = {k: v for k, v in s1_tr.items() if k in train_ids}

    # ── 4. Generate training candidates ───────────────────────────
    log.info("\n[4/8] Blocking — training candidates …")
    train_candidates = generate_candidates(s1_tr_split, s2_tr, s3_tr, "train")

    # ── 5. Build training pairs & feature matrix ───────────────────
    log.info("\n[5/8] Building training feature matrix …")
    cache_key_feat = "train_features"
    cached_feat = load_cache(cache_key_feat)

    if cached_feat:
        X_train, y_train, X_val, y_val, val_pair_ids = cached_feat
        log.info("  loaded feature matrices from cache")
    else:
        # Training pairs from train split
        train_raw_pairs = build_train_pairs(s1_tr_split, train_candidates, gt)
        X_train, y_train, _ = build_feature_matrix(train_raw_pairs, s1_tr, sx_tr)
        log.info(f"  Train: X={X_train.shape}  pos={y_train.sum():,}")

        # Validation pairs — need to block val S1 against S2/S3
        s1_val_split = {k: v for k, v in s1_tr.items() if k in val_ids}
        val_candidates = generate_candidates(s1_val_split, s2_tr, s3_tr, "val")

        val_raw_pairs = build_train_pairs(s1_val_split, val_candidates, gt, neg_ratio=10)
        X_val, y_val, val_pair_ids = build_feature_matrix(val_raw_pairs, s1_tr, sx_tr)
        log.info(f"  Val:   X={X_val.shape}   pos={y_val.sum():,}")

        save_cache((X_train, y_train, X_val, y_val, val_pair_ids), cache_key_feat)

    # ── 6. Train LightGBM ─────────────────────────────────────────
    log.info("\n[6/8] Training LightGBM …")
    model_cache = load_cache("lgbm_model")
    if model_cache:
        model = model_cache
        log.info("  loaded model from cache")
    else:
        model = train_model(X_train, y_train)
        save_cache(model, "lgbm_model")

    # ── 7. Tune threshold on validation set ───────────────────────
    log.info("\n[7/8] Tuning threshold on validation split …")
    val_proba = model.predict_proba(X_val)[:, 1]
    val_s1_ids = list({s1_id for s1_id, _ in val_pair_ids})
    threshold = tune_threshold(val_s1_ids, val_pair_ids, val_proba, gt)

    # ── 8. Full test pass ─────────────────────────────────────────
    log.info("\n[8/8] Processing test data …")
    s1_te, s2_te, s3_te = load_and_normalize("test")
    sx_te = {**s2_te, **s3_te}

    test_candidates = generate_candidates(s1_te, s2_te, s3_te, "test")

    matching, cand_out = predict(s1_te, sx_te, test_candidates, model, threshold)

    # Serialise cand_out to string lists
    cand_str_out = {s1_id: list(ids) for s1_id, ids in cand_out.items()}

    write_outputs(matching, cand_str_out, list(s1_te.keys()))

    elapsed = time.time() - t_start
    log.info(f"\nDone! Total time: {elapsed/60:.1f} min")
    log.info(f"Outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
