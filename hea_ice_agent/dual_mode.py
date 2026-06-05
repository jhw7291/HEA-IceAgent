#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEA-IceAgent v2.2 — Dual-Mode: SCREEN + REASON

SCREEN mode:  Ranks 218 compositions FROM the DFT database using
              DFT-verified lattice constants + full 2D matching.

REASON mode:  Enumerates ALL possible HEA compositions in the
              8-element space (Al-Si-Cr-Mn-Fe-Co-Ni-Cu) with
              variant stoichiometries, predicts their lattice
              constants via RandomForest ML, and scores them
              for ice nucleation control.

Output:
  results/candidates_promoter.csv     — SCREEN: top 50 from DB
  results/candidates_inhibitor.csv    — SCREEN: top 50 from DB
  results/candidates_iri.csv          — SCREEN: top 50 from DB
  results/candidates_all_tasks.csv    — SCREEN: all 218 ranked
  results/reasoned_new_candidates.csv — REASON: 1170 predicted
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor

from .config import (
    RESULTS_DIR, DATA_RAW_DIR, FILE_HEA_MAIN,
    ALL_ELEMS := ['Al','Si','Cr','Mn','Fe','Co','Ni','Cu'],
    ATOMIC_RADII, ELECTRONEGATIVITY,
)
from .loader import load_and_merge, clean_dataset, get_composition_groups
from .lattice_matching import load_dft_lattice_constants
from .utils import compute_composition_from_formula, configurational_entropy, atomic_radius_mismatch


ALL_ELEMS = ['Al','Si','Cr','Mn','Fe','Co','Ni','Cu']


def build_ml_features(chem_sys: str) -> dict:
    """Build ML feature vector from a chemical system string."""
    comp = compute_composition_from_formula(chem_sys)
    feats = {f'frac_{e}': comp.get(e, 0.0) for e in ALL_ELEMS}
    feats['S_conf'] = configurational_entropy(comp)
    feats['delta_r'] = atomic_radius_mismatch(comp, ATOMIC_RADII)
    feats['n_elem'] = sum(1 for v in comp.values() if v > 0.001)
    en_vals = [ELECTRONEGATIVITY.get(e, 1.8) for e in comp if comp.get(e, 0) > 0.001]
    feats['avg_EN'] = np.mean(en_vals) if en_vals else 1.8
    feats['var_EN'] = np.var(en_vals) if en_vals else 0.0
    return feats


def train_ml_models(groups: pd.DataFrame, dft_a: pd.DataFrame):
    """Train RandomForest models for BCC/FCC lattice constants and formation energy."""
    X_data = []
    for _, row in groups.iterrows():
        feats = build_ml_features(row['chemical_system'])
        X_data.append({'chemical_system': row['chemical_system'], **feats})
    X_all = pd.DataFrame(X_data)
    feat_cols = [c for c in X_all.columns if c != 'chemical_system']

    bcc_m = X_all.merge(
        dft_a[dft_a['lattice'] == 'bcc'][['chemical_system', 'a_dft_mean']],
        on='chemical_system', how='inner')
    fcc_m = X_all.merge(
        dft_a[dft_a['lattice'] == 'fcc'][['chemical_system', 'a_dft_mean']],
        on='chemical_system', how='inner')

    rf_bcc = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    rf_bcc.fit(bcc_m[feat_cols], bcc_m['a_dft_mean'])
    rf_fcc = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    rf_fcc.fit(fcc_m[feat_cols], fcc_m['a_dft_mean'])
    rf_ef = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    rf_ef.fit(X_all[feat_cols], groups['mean_Ef'])

    return rf_bcc, rf_fcc, rf_ef, feat_cols


def enumerate_candidates(existing_set: set) -> list:
    """Enumerate all element combos (3-7 elements) with equimolar +/- 1 variants."""
    candidates = []
    for n_elem in range(3, 8):
        for elem_combo in combinations(ALL_ELEMS, n_elem):
            elems = list(elem_combo)
            for variant in range(n_elem + 1):
                stoich = {e: 1 for e in elems}
                if variant > 0:
                    stoich[elems[variant - 1]] = 2
                parts = [f'{e}{c}' for e, c in sorted(stoich.items())]
                chem_sys = ''.join(parts)
                feats = build_ml_features(chem_sys)
                candidates.append({
                    'chemical_system': chem_sys,
                    'n_elem': n_elem,
                    'is_new': chem_sys not in existing_set,
                    'feats': feats,
                })
    return candidates


def predict_candidates(candidates: list, rf_bcc, rf_fcc, rf_ef, feat_cols: list):
    """Predict BCC/FCC lattice constants and formation energy for all candidates."""
    feat_df = pd.DataFrame([c['feats'] for c in candidates])
    pb = rf_bcc.predict(feat_df[feat_cols])
    pf = rf_fcc.predict(feat_df[feat_cols])
    pe = rf_ef.predict(feat_df[feat_cols])
    for i, c in enumerate(candidates):
        c['pred_bcc_a'] = float(pb[i])
        c['pred_fcc_a'] = float(pf[i])
        c['pred_Ef'] = float(pe[i])
    return candidates


def score_candidates(candidates: list, db_promoter_csv: str):
    """Score candidates for all three tasks using proxy metrics."""
    db_prom = pd.read_csv(db_promoter_csv)
    bcc_good = db_prom[db_prom['lattice_type'] == 'bcc']
    fcc_good = db_prom[db_prom['lattice_type'] == 'fcc']
    opt_bcc = float(bcc_good['a_dft'].mean()) if len(bcc_good) > 0 else 4.9
    opt_fcc = float(fcc_good['a_dft'].mean()) if len(fcc_good) > 0 else 9.1

    agi = 0.01327
    for c in candidates:
        d_bcc = abs(c['pred_bcc_a'] - opt_bcc) / opt_bcc * 0.1
        d_fcc = abs(c['pred_fcc_a'] - opt_fcc) / opt_fcc * 0.1
        c['best_lat'] = 'bcc' if d_bcc < d_fcc else 'fcc'
        c['est_mm'] = min(d_bcc, d_fcc)
        # Task A: promoter
        c['promoter_score'] = float(np.exp(-(c['est_mm'] - agi)**2 / (2 * 0.03**2)))
        # Task B: inhibitor
        c['inhibitor_score'] = float(c['est_mm'])
        # Task C: IRI
        c['iri_score'] = float(
            0.3 * c['feats']['var_EN'] * 100 +
            0.25 * c['feats']['S_conf'] +
            0.25 * c['feats']['delta_r']
        )
    return candidates


def run_dual_mode(n_jobs: int = 1, verbose: bool = True):
    """Run both SCREEN and REASON modes.

    SCREEN:  Run the original pipeline (Phase 1-5) for database ranking.
    REASON:  Train ML, enumerate all possible compositions, predict, rank.
    """
    t0 = time.time()

    if verbose:
        print("\n" + "=" * 72)
        print("  HEA-IceAgent v2.2 — DUAL MODE: SCREEN + REASON")
        print("=" * 72)

    # ── SCREEN mode ──
    if verbose:
        print("\n--- MODE 1: SCREEN (Database Ranking) ---")
    from .pipeline import HEAIcePipeline
    pipeline = HEAIcePipeline(n_jobs=n_jobs, verbose=verbose)
    outputs = pipeline.run_full(skip_download=True)

    # ── REASON mode ──
    if verbose:
        print("\n" + "=" * 72)
        print("--- MODE 2: REASON (ML Inference for New Compositions) ---")

    df = clean_dataset(load_and_merge())
    groups = get_composition_groups(df)
    dft_a = load_dft_lattice_constants()
    existing_set = set(groups['chemical_system'])

    # Train
    rf_bcc, rf_fcc, rf_ef, feat_cols = train_ml_models(groups, dft_a)
    # Enumerate
    candidates = enumerate_candidates(existing_set)
    new = [c for c in candidates if c['is_new']]
    if verbose:
        print(f"  ML models trained on {len(groups)} compositions")
        print(f"  Enumerated {len(candidates)} candidates ({len(new)} new)")

    # Predict
    candidates = predict_candidates(candidates, rf_bcc, rf_fcc, rf_ef, feat_cols)

    # Score
    promoters_csv = os.path.join(RESULTS_DIR, 'candidates_promoter.csv')
    candidates = score_candidates(new, promoters_csv)

    # Save reasoned
    reasoned_path = os.path.join(RESULTS_DIR, 'reasoned_new_candidates.csv')
    df_reasoned = pd.DataFrame(new)
    keep_cols = ['chemical_system', 'n_elem', 'best_lat', 'pred_bcc_a',
                 'pred_fcc_a', 'pred_Ef', 'est_mm', 'promoter_score',
                 'inhibitor_score', 'iri_score']
    save_cols = [c for c in keep_cols if c in df_reasoned.columns]
    df_reasoned[save_cols].to_csv(reasoned_path, index=False)
    outputs['reasoned_new_candidates.csv'] = reasoned_path

    # Print top results
    if verbose:
        _print_dual_results(new)

    if verbose:
        print(f"\n  Dual mode complete in {time.time() - t0:.0f}s")
        print(f"  SCREEN: {len(groups)} DB compositions ranked")
        print(f"  REASON: {len(new)} new compositions predicted")
        print(f"  All output in: {RESULTS_DIR}")

    return outputs


def _print_dual_results(new_candidates: list):
    """Print top candidates from both modes."""
    results_dir = RESULTS_DIR
    db_prom = pd.read_csv(os.path.join(results_dir, 'candidates_promoter.csv'))
    db_inhib = pd.read_csv(os.path.join(results_dir, 'candidates_inhibitor.csv'))
    db_iri = pd.read_csv(os.path.join(results_dir, 'candidates_iri.csv'))

    new_prom = sorted(new_candidates, key=lambda x: x['promoter_score'], reverse=True)
    new_inhib = sorted(new_candidates, key=lambda x: x['inhibitor_score'], reverse=True)
    new_iri = sorted(new_candidates, key=lambda x: x['iri_score'], reverse=True)

    print()
    print("  === SCREEN: Top-5 DB Promoters ===")
    for i, (_, r) in enumerate(db_prom.head(5).iterrows()):
        lt = str(r['lattice_type'])
        print(f"  {i+1}. {r['chemical_system']:<35s} {lt:4s} d={r['best_mismatch']*100:.3f}%")

    print()
    print("  === REASON: Top-10 Predicted NEW Promoters ===")
    for i, c in enumerate(new_prom[:10]):
        a = c[f'pred_{c[\"best_lat\"]}_a']
        print(f"  {i+1:2d}. {c['chemical_system']:<35s} {c['best_lat']:3s} a={a:.3f}A Ef={c['pred_Ef']:+.3f}")

    print()
    print("  === REASON: Top-5 Predicted NEW Inhibitors ===")
    for i, c in enumerate(new_inhib[:5]):
        print(f"  {i+1}. {c['chemical_system']:<35s} {c['best_lat']:3s} a={c[f'pred_{c[\"best_lat\"]}_a']:.3f}A")

    print()
    print("  === REASON: Top-5 Predicted NEW IRI ===")
    for i, c in enumerate(new_iri[:5]):
        print(f"  {i+1}. {c['chemical_system']:<35s} S_conf={c['feats']['S_conf']:.3f} IRI={c['iri_score']:.3f}")


if __name__ == "__main__":
    run_dual_mode(n_jobs=4)
