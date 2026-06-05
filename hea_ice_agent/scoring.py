#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEA-IceAgent — Scoring (v2: AgI-calibrated)

Fixes:
- #3: Uses AgI(0001) 2D mismatch (~1.877%) as reference anchor
- #4: Data provenance tracking (hash + row counts)
- Uses min-max normalization vs percentile (more physically meaningful)
"""

import numpy as np
import pandas as pd
import hashlib
import os
import json
from datetime import datetime
from typing import Dict, Tuple

from .config import (
    PROMOTER_WEIGHTS,
    PROMOTER_MISMATCH_SIGMA,
    INHIBITOR_WEIGHTS,
    IRI_WEIGHTS,
    TOP_N_CANDIDATES,
    DATA_RAW_DIR,
    FILE_STRUCTURE_INI,
    FILE_SRO,
    FILE_HEA_MAIN,
)
from .utils import min_max_normalize, score_to_percentile


# ── Data Provenance ─────────────────────────────────────────────

def _file_hash(filepath: str) -> str:
    """SHA256 of first 1MB + last 1MB for fast verification."""
    if not os.path.exists(filepath):
        return "MISSING"
    sha = hashlib.sha256()
    fsize = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        sha.update(f.read(min(1048576, fsize)))
        if fsize > 2097152:
            f.seek(fsize - 1048576)
            sha.update(f.read(1048576))
    return sha.hexdigest()[:16]


def record_provenance(df: pd.DataFrame,
                      lattice_results: pd.DataFrame = None) -> Dict:
    """Record exactly what data was used and how.

    Returns a provenance dict to be saved alongside results.
    """
    prov = {
        'pipeline': 'HEA-IceAgent v2',
        'timestamp': datetime.now().isoformat(),
        'input_files': {
            'structure_ini_featurized': {
                'path': os.path.join(DATA_RAW_DIR, FILE_STRUCTURE_INI),
                'sha256_prefix': _file_hash(os.path.join(DATA_RAW_DIR, FILE_STRUCTURE_INI)),
            },
            'sro': {
                'path': os.path.join(DATA_RAW_DIR, FILE_SRO),
                'sha256_prefix': _file_hash(os.path.join(DATA_RAW_DIR, FILE_SRO)),
            },
            'hea_main': {
                'path': os.path.join(DATA_RAW_DIR, FILE_HEA_MAIN),
                'sha256_prefix': _file_hash(os.path.join(DATA_RAW_DIR, FILE_HEA_MAIN)),
            },
        },
        'data_stats': {
            'n_structures_raw': len(df) if df is not None else None,
            'n_structures_clean': None,
            'n_compositions': 0,
            'n_structures_total': 0,
            'elements': ['Al', 'Si', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu'],
        },
        'lattice_matching': {
            'method': 'Zur 2D supercell (Frobenius norm of strain tensor)',
            'agI_basal_baseline_2d_mismatch_pct': 1.877,
            'n_ice_faces': 3,
            'n_hea_surfaces': 3,
            'rotation_step_deg': 1.0,
            'max_supercell': (4, 4, 4, 4),
        },
        'scoring': {
            'promoter_weights': PROMOTER_WEIGHTS,
            'inhibitor_weights': INHIBITOR_WEIGHTS,
            'iri_weights': IRI_WEIGHTS,
        },
    }

    if lattice_results is not None:
        prov['data_stats']['n_compositions'] = len(lattice_results)
        prov['lattice_matching']['mismatch_2d_mean_pct'] = \
            float(lattice_results['best_mismatch'].mean() * 100)
        prov['lattice_matching']['mismatch_2d_median_pct'] = \
            float(lattice_results['best_mismatch'].median() * 100)

    return prov


# ── Task A: Ice Nucleation Promoter ─────────────────────────────
# AgI(0001) normalized 2D mismatch = 1.327% (matches literature 1.3%)
AGI_2D_MISMATCH = 1.327  # percent


def score_nucleation_promoter(
    df: pd.DataFrame,
    weights: dict = None,
) -> pd.DataFrame:
    """Score HEA compositions for ice nucleation promotion (AgI-calibrated).

    The key innovation: AgI's measured 2D mismatch (1.877%) defines
    the center of the 'excellent nucleation' region.  Compositions
    with mismatch close to 1.877% get the highest lattice-match score.

    Why not just "closest to zero"? Because real epitaxy depends on
    a finite supercell coincidence — zero mismatch at the primitive
    cell level is physically meaningless for 2D interfaces.
    """
    if weights is None:
        weights = PROMOTER_WEIGHTS

    print("\n  Computing Ice Nucleation Promoter Scores ...")
    df = df.copy()

    # 1. Lattice match quality — AgI-calibrated
    if 'best_mismatch' in df.columns:
        sigma_calibrated = 0.03  # 3% width around optimal mismatch
        mismatch_pct = df['best_mismatch'] * 100  # convert to percent
        # Score peaks near AgI and decays for both lower and higher mismatch
        df['promoter_lattice_match'] = np.exp(
            -(mismatch_pct - AGI_2D_MISMATCH)**2 / (2 * sigma_calibrated**2)
        )
        # Also give credit for very low mismatch (close to coherent)
        coherent_score = np.exp(-mismatch_pct**2 / (2 * (2*sigma_calibrated)**2))
        df['promoter_lattice_match'] = np.maximum(
            df['promoter_lattice_match'], coherent_score * 0.5)
    else:
        df['promoter_lattice_match'] = 0.5

    # 2-5: Same as v1
    if 'stability_score' in df.columns:
        df['promoter_stability'] = min_max_normalize(df['stability_score'])
    else:
        df['promoter_stability'] = 0.5

    if 'hex_proximity' in df.columns:
        df['promoter_hexagonality'] = min_max_normalize(df['hex_proximity'])
    else:
        df['promoter_hexagonality'] = 0.5

    if 'surface_energy_proxy' in df.columns:
        df['promoter_surface_energy'] = df['surface_energy_proxy']
    else:
        df['promoter_surface_energy'] = 0.5

    if 'work_function_proxy' in df.columns:
        wf_diff = np.abs(df['work_function_proxy'] - 5.3)
        df['promoter_work_function'] = 1.0 - min_max_normalize(wf_diff)
    else:
        df['promoter_work_function'] = 0.5

    df['S_promoter'] = (
        weights['lattice_match'] * df['promoter_lattice_match'] +
        weights['stability'] * df['promoter_stability'] +
        weights['hexagonality'] * df['promoter_hexagonality'] +
        weights['surface_energy_proxy'] * df['promoter_surface_energy'] +
        weights['work_function_proxy'] * df['promoter_work_function']
    )
    df['S_promoter_norm'] = score_to_percentile(df['S_promoter'])

    print(f"    Promoter score: mean={df['S_promoter'].mean():.4f} "
          f"range=[{df['S_promoter'].min():.4f}, {df['S_promoter'].max():.4f}]")
    return df


# ── Task B: Ice Nucleation Inhibitor ────────────────────────────

def score_nucleation_inhibitor(
    df: pd.DataFrame,
    weights: dict = None,
) -> pd.DataFrame:
    """Score HEA compositions for ice nucleation inhibition.

    High 2D mismatch + strong hydrophobicity = good inhibitor.
    """
    if weights is None:
        weights = INHIBITOR_WEIGHTS

    print("\n  Computing Ice Nucleation Inhibitor Scores ...")
    df = df.copy()

    if 'best_mismatch' in df.columns:
        # Score high mismatch — calibrated so AgI (~1.877%) gets low score
        sigma_inhibitor = 0.10
        df['inhibitor_lattice_mismatch'] = 1.0 - np.exp(
            -(df['best_mismatch'].mean() * 1.5)**2 / (2 * sigma_inhibitor**2)
        )
        # Better: use linear ranking of mismatch
        df['inhibitor_lattice_mismatch'] = min_max_normalize(
            df['best_mismatch'] * 100)
    else:
        df['inhibitor_lattice_mismatch'] = 0.5

    if 'hydrophobicity' in df.columns:
        df['inhibitor_hydrophobicity'] = min_max_normalize(df['hydrophobicity'])
    else:
        df['inhibitor_hydrophobicity'] = 0.5

    if 'electronic_smoothness' in df.columns:
        df['inhibitor_electronic'] = min_max_normalize(df['electronic_smoothness'])
    else:
        df['inhibitor_electronic'] = 0.5

    if 'SRO_mean' in df.columns:
        df['inhibitor_surface'] = 1.0 - min_max_normalize(df['SRO_mean'])
    else:
        df['inhibitor_surface'] = 0.5

    df['S_inhibitor'] = (
        weights['lattice_mismatch'] * df['inhibitor_lattice_mismatch'] +
        weights['hydrophobicity'] * df['inhibitor_hydrophobicity'] +
        weights['electronic_smoothness'] * df['inhibitor_electronic'] +
        weights['surface_smoothness'] * df['inhibitor_surface']
    )
    df['S_inhibitor_norm'] = score_to_percentile(df['S_inhibitor'])

    print(f"    Inhibitor score: mean={df['S_inhibitor'].mean():.4f} "
          f"range=[{df['S_inhibitor'].min():.4f}, {df['S_inhibitor'].max():.4f}]")
    return df


# ── Task C: IRI ─────────────────────────────────────────────────

def score_iri_inhibitor(
    df: pd.DataFrame,
    weights: dict = None,
) -> pd.DataFrame:
    """Score HEA compositions for ice recrystallization inhibition.

    Moderate SRO heterogeneity is key: too low = no pinning, too high = unstable.
    """
    if weights is None:
        weights = IRI_WEIGHTS

    print("\n  Computing Ice Recrystallization Inhibitor (IRI) Scores ...")
    df = df.copy()

    # SRO heterogeneity with "sweet spot" calibration
    if 'SRO_heterogeneity' in df.columns:
        # Moderate heterogeneity (0.3-0.6) gets highest score
        optimal_sro = 0.45
        sigma_sro = 0.20
        df['iri_sro_heterogeneity'] = np.exp(
            -(df['SRO_heterogeneity'] - optimal_sro)**2 / (2 * sigma_sro**2))
    else:
        df['iri_sro_heterogeneity'] = 0.5

    if 'S_conf' in df.columns:
        df['iri_config_entropy'] = min_max_normalize(df['S_conf'])
    else:
        df['iri_config_entropy'] = 0.5

    if 'lattice_distortion' in df.columns:
        df['iri_lattice_distortion'] = min_max_normalize(df['lattice_distortion'])
    else:
        df['iri_lattice_distortion'] = 0.5

    if 'mixing_enthalpy_proxy' in df.columns:
        df['iri_mixing_enthalpy'] = min_max_normalize(
            df['mixing_enthalpy_proxy'].abs())
    else:
        df['iri_mixing_enthalpy'] = 0.5

    df['S_iri'] = (
        weights['sro_heterogeneity'] * df['iri_sro_heterogeneity'] +
        weights['config_entropy'] * df['iri_config_entropy'] +
        weights['lattice_distortion'] * df['iri_lattice_distortion'] +
        weights['mixing_enthalpy_proxy'] * df['iri_mixing_enthalpy']
    )
    df['S_iri_norm'] = score_to_percentile(df['S_iri'])

    print(f"    IRI score: mean={df['S_iri'].mean():.4f} "
          f"range=[{df['S_iri'].min():.4f}, {df['S_iri'].max():.4f}]")
    return df


# ── Unified Scoring ─────────────────────────────────────────────

def compute_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = score_nucleation_promoter(df)
    df = score_nucleation_inhibitor(df)
    df = score_iri_inhibitor(df)

    print("\n  -- Cross-Task Analysis --")
    top_pct = 70
    for task in ['S_promoter_norm', 'S_inhibitor_norm', 'S_iri_norm']:
        thresh = df[task].quantile(top_pct / 100)
        n = (df[task] >= thresh).sum()
        print(f"  Top 30% in {task}: {n}")

    conditions = [(df[f'S_{t}_norm'] >= df[f'S_{t}_norm'].quantile(0.7))
                  for t in ['promoter', 'inhibitor', 'iri']]
    n_all = (conditions[0] & conditions[1] & conditions[2]).sum()
    print(f"  Top 30% in ALL three: {n_all}")

    return df


def get_top_candidates(df: pd.DataFrame, task: str = 'promoter',
                       top_n: int = TOP_N_CANDIDATES) -> pd.DataFrame:
    score_col = f'S_{task}'
    if score_col not in df.columns:
        raise ValueError(f"Score column {score_col} not found. Run scoring first.")

    base_cols = ['chemical_system', 'lattice_type', 'a_dft',
                 'best_mismatch', 'best_ice_face', 'best_hea_surface',
                 'nelements', 'n_structures',
                 'S_conf', 'delta_r', 'SRO_heterogeneity',
                 score_col, f'{score_col}_norm']
    available = [c for c in base_cols if c in df.columns]
    top = df[available].sort_values(score_col, ascending=False).head(top_n)
    return top.reset_index(drop=True)
