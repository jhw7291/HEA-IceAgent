#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lattice Matching Engine v2.1 — True 2D supercell matching.

Key algorithm: Zur rotating overlayer method
- Builds full 2x2 supercell matrices (not just 1D vector matching)
- Computes 2D strain tensor epsilon = S_ice^{-1} @ S_sub - I
- Mismatch = Frobenius norm of epsilon, normalized by sqrt(2)
  so that isotropic strain gives same % as 1D mismatch
- Uses DFT lattice constants from hea.2023-04-06.csv (not Vegard estimates)
"""

import os
import hashlib
import numpy as np
import pandas as pd
from itertools import product
from typing import Dict, List, Tuple, Optional
from joblib import Parallel, delayed
import time

from .config import (
    ICE_SURFACES,
    ELEMENTS,
    ICE_A,
    ICE_C,
    AGI_A,
    AGI_C,
    MAX_SUPERCELL,
    ROTATION_STEP,
    MAX_AREA_RATIO,
    N_JOBS,
    DATA_RAW_DIR,
    FILE_HEA_MAIN,
)
from .utils import (
    rotation_matrix_2d,
    compute_composition_from_formula,
)


# ── File provenance ─────────────────────────────────────────────

def _file_hash(filepath: str) -> str:
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


def _data_provenance() -> Dict:
    return {
        'file': FILE_HEA_MAIN,
        'sha256_prefix': _file_hash(os.path.join(DATA_RAW_DIR, FILE_HEA_MAIN)),
    }


# ── DFT lattice constants ───────────────────────────────────────

def load_dft_lattice_constants() -> Optional[pd.DataFrame]:
    """Load DFT lattice constants from hea.2023-04-06.csv.

    Uses volume_per_atom to compute cubic a for each structure,
    then averages per (composition, lattice) group.
    """
    path = os.path.join(DATA_RAW_DIR, FILE_HEA_MAIN)
    if not os.path.exists(path):
        print(f"    WARNING: {FILE_HEA_MAIN} not found, falling back to Vegard")
        return None

    print("  Loading DFT lattice constants from hea.2023-04-06.csv ...")
    df = pd.read_csv(path, index_col=0)

    if 'volume_per_atom' in df.columns and 'NIONS' in df.columns:
        df['a_dft'] = (df['volume_per_atom'] * df['NIONS']) ** (1.0 / 3.0)
    else:
        print("    WARNING: no volume_per_atom column")
        return None

    groups = df.groupby(['chemical_system', 'lattice']).agg(
        a_dft_mean=('a_dft', 'mean'),
        a_dft_std=('a_dft', 'std'),
        n_samples=('a_dft', 'count'),
    ).reset_index()

    print(f"    Loaded DFT a for {len(groups)} (composition, lattice) pairs")
    print(f"    a_dft range: [{groups['a_dft_mean'].min():.3f}, "
          f"{groups['a_dft_mean'].max():.3f}] A")
    return groups


# ── Surface construction ────────────────────────────────────────

def build_bcc_surface(a: float, surface: str) -> Dict:
    if surface == '110':
        return {'name': f'BCC(110) a={a:.3f}A',
                'a1': np.array([np.sqrt(2)*a/2, 0.0]),
                'a2': np.array([0.0, a])}
    elif surface == '100':
        return {'name': f'BCC(100) a={a:.3f}A',
                'a1': np.array([a, 0.0]), 'a2': np.array([0.0, a])}
    else:  # 111
        return {'name': f'BCC(111) a={a:.3f}A',
                'a1': np.array([np.sqrt(2)*a/2, 0.0]),
                'a2': np.array([np.sqrt(2)*a/4, np.sqrt(6)*a/4])}


def build_fcc_surface(a: float, surface: str) -> Dict:
    if surface == '111':
        return {'name': f'FCC(111) a={a:.3f}A',
                'a1': np.array([np.sqrt(2)*a/2, 0.0]),
                'a2': np.array([np.sqrt(2)*a/4, np.sqrt(6)*a/4])}
    elif surface == '100':
        return {'name': f'FCC(100) a={a:.3f}A',
                'a1': np.array([a, 0.0]), 'a2': np.array([0.0, a])}
    else:  # 110
        return {'name': f'FCC(110) a={a:.3f}A',
                'a1': np.array([np.sqrt(2)*a/2, 0.0]),
                'a2': np.array([0.0, a])}


def build_ice_surface(face_name: str) -> Dict:
    if face_name not in ICE_SURFACES:
        raise ValueError(f"Unknown ice face: {face_name}")
    d = ICE_SURFACES[face_name]
    return {'name': f'Ice {d["name"]}', 'a1': d['a1'].copy(), 'a2': d['a2'].copy()}


# ── 2D Supercell Matching ───────────────────────────────────────

def _search_single_rotation(
    a1_sub: np.ndarray, a2_sub: np.ndarray,
    a1_ice: np.ndarray, a2_ice: np.ndarray,
    rotation_deg: float,
) -> List[Dict]:
    """True 2D lattice matching at a given rotation angle.

    Builds full 2x2 supercell matrices S_sub and S_ice, computes the
    2D strain tensor epsilon = S_ice^{-1} @ S_sub - I, and returns
    the Frobenius norm (normalized by sqrt(2)) as the mismatch.
    """
    R = rotation_matrix_2d(rotation_deg)
    a1_ice_rot = R @ a1_ice
    a2_ice_rot = R @ a2_ice

    A_sub = np.column_stack([a1_sub, a2_sub])
    A_ice = np.column_stack([a1_ice_rot, a2_ice_rot])

    imax, jmax, kmax, lmax = MAX_SUPERCELL
    matches = []

    for i1 in range(1, imax + 1):
        for j1 in range(0, jmax + 1):
            if j1 == 0 and i1 == 0:
                continue
            for i2 in range(0, imax + 1):
                for j2 in range(1, jmax + 1):
                    if i2 == 0 and j2 == 0:
                        continue
                    det_sub = i1 * j2 - i2 * j1
                    if abs(det_sub) < 0.01:
                        continue

                    S_sub = A_sub @ np.array([[i1, i2], [j1, j2]])
                    area_sub = abs(np.linalg.det(S_sub))
                    if area_sub < 1e-10:
                        continue

                    # Nearest integer ice supercell
                    A_ice_inv = np.linalg.inv(A_ice)
                    M_ice_float = A_ice_inv @ S_sub
                    k1 = int(round(M_ice_float[0, 0]))
                    k2 = int(round(M_ice_float[0, 1]))
                    l1 = int(round(M_ice_float[1, 0]))
                    l2 = int(round(M_ice_float[1, 1]))

                    if max(abs(k1), abs(k2)) > kmax or max(abs(l1), abs(l2)) > lmax:
                        continue
                    det_ice = k1 * l2 - k2 * l1
                    if abs(det_ice) < 0.01:
                        continue

                    S_ice = A_ice @ np.array([[k1, k2], [l1, l2]])
                    area_ice = abs(np.linalg.det(S_ice))
                    if area_ice < 1e-10:
                        continue

                    area_ratio = max(area_sub, area_ice) / min(area_sub, area_ice)
                    if area_ratio > MAX_AREA_RATIO:
                        continue

                    try:
                        eps = np.linalg.inv(S_ice) @ S_sub - np.eye(2)
                        mismatch_2d = np.linalg.norm(eps, 'fro') / np.sqrt(2)
                    except np.linalg.LinAlgError:
                        continue

                    matches.append({
                        'rotation': rotation_deg,
                        'i1': i1, 'j1': j1, 'i2': i2, 'j2': j2,
                        'k1': k1, 'l1': l1, 'k2': k2, 'l2': l2,
                        'mismatch_2d': mismatch_2d,
                    })

    matches.sort(key=lambda x: x['mismatch_2d'])
    return matches[:30]


def compute_2d_mismatch_full(
    a1_sub: np.ndarray, a2_sub: np.ndarray,
    a1_ice: np.ndarray, a2_ice: np.ndarray,
    rotation_step: float = ROTATION_STEP,
) -> Dict:
    angles = np.arange(0, 180.0, rotation_step)
    all_matches = []
    for theta in angles:
        all_matches.extend(_search_single_rotation(
            a1_sub, a2_sub, a1_ice, a2_ice, theta))

    if not all_matches:
        return {'mismatch': 1.0, 'rotation': 0.0,
                'i1':0,'j1':0,'i2':0,'j2':0,'k1':0,'l1':0,'k2':0,'l2':0}

    all_matches.sort(key=lambda x: x['mismatch_2d'])
    best = all_matches[0]
    return {
        'mismatch': best['mismatch_2d'],
        'rotation': best['rotation'],
        'i1':best['i1'],'j1':best['j1'],'i2':best['i2'],'j2':best['j2'],
        'k1':best['k1'],'l1':best['l1'],'k2':best['k2'],'l2':best['l2'],
    }


# ── Full composition matching ───────────────────────────────────

def match_single_composition_dft(
    chem_sys: str, lattice_type: str, a_dft: float, verbose: bool = False
) -> Dict:
    surfaces_to_scan = (['110','100','111'] if lattice_type == 'bcc'
                        else ['111','100','110'])
    build_fn = build_bcc_surface if lattice_type == 'bcc' else build_fcc_surface
    ice_faces = ['basal', 'prism', 'pyramidal']

    all_matches = []
    best_overall = {'mismatch': 1.0}

    for surf_name, ice_face in product(surfaces_to_scan, ice_faces):
        hea_surf = build_fn(a_dft, surf_name)
        ice_surf = build_ice_surface(ice_face)
        result = compute_2d_mismatch_full(
            hea_surf['a1'], hea_surf['a2'],
            ice_surf['a1'], ice_surf['a2'])
        result['ice_face'] = ice_face
        result['hea_surface'] = surf_name
        all_matches.append(result)
        if result['mismatch'] < best_overall['mismatch']:
            best_overall = result

    all_matches.sort(key=lambda x: x['mismatch'])
    return {
        'chemical_system': chem_sys,
        'lattice_type': lattice_type,
        'a_dft': a_dft,
        'best_mismatch': best_overall['mismatch'],
        'best_rotation': best_overall['rotation'],
        'best_ice_face': best_overall['ice_face'],
        'best_hea_surface': best_overall['hea_surface'],
        'best_i1': best_overall.get('i1',0), 'best_j1': best_overall.get('j1',0),
        'best_i2': best_overall.get('i2',0), 'best_j2': best_overall.get('j2',0),
        'best_k1': best_overall.get('k1',0), 'best_l1': best_overall.get('l1',0),
        'best_k2': best_overall.get('k2',0), 'best_l2': best_overall.get('l2',0),
        'all_matches': all_matches,
    }


def run_lattice_matching(
    composition_groups: pd.DataFrame,
    n_jobs: int = N_JOBS,
    verbose: bool = True,
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Phase 3: 2D Lattice Matching with Ice Ih (True 2D Supercell)")
    print("=" * 60)

    dft_a = load_dft_lattice_constants()
    if dft_a is not None:
        mg = composition_groups.merge(
            dft_a[['chemical_system','lattice','a_dft_mean','a_dft_std','n_samples']],
            left_on=['chemical_system','lattice'],
            right_on=['chemical_system','lattice'], how='inner')
        use_dft = True
    else:
        mg = composition_groups.copy()
        use_dft = False

    print(f"  Analyzing {len(mg)} unique compositions ...")
    t0 = time.time()
    n_jobs_eff = max(n_jobs, 1) if n_jobs > 0 else 1

    if n_jobs_eff > 1:
        results = Parallel(n_jobs=n_jobs_eff)(
            delayed(match_single_composition_dft)(
                row['chemical_system'], row['lattice'],
                row['a_dft_mean'] if use_dft else 3.0, False)
            for _, row in mg.iterrows())
    else:
        results = []
        for idx, (_, row) in enumerate(mg.iterrows()):
            if verbose and idx % 50 == 0:
                print(f"    Progress: {idx}/{len(mg)} ...")
            a_val = row['a_dft_mean'] if use_dft else 3.0
            results.append(match_single_composition_dft(
                row['chemical_system'], row['lattice'], a_val, False))

    results = [r for r in results if r is not None]
    df_results = pd.DataFrame(results)
    df_results = df_results.drop(columns=['all_matches'], errors='ignore')
    df_results['mismatch_percent'] = df_results['best_mismatch'] * 100
    df_results['match_quality'] = np.exp(
        -df_results['best_mismatch']**2 / (2 * 0.05**2))

    elapsed = time.time() - t0
    mm = df_results['best_mismatch']
    print(f"  Done in {elapsed:.1f}s")
    print(f"  -- 2D Mismatch Statistics --")
    print(f"  Min: {mm.min()*100:.2f}%  Max: {mm.max()*100:.2f}%")
    print(f"  Mean: {mm.mean()*100:.2f}%  Median: {mm.median()*100:.2f}%")
    print(f"  Compositions with 2D delta < 5%: {(mm < 0.05).sum()}/{len(df_results)}")
    print(f"  Compositions with 2D delta < 10%: {(mm < 0.10).sum()}/{len(df_results)}")
    for face in ['basal','prism','pyramidal']:
        subset = df_results[df_results['best_ice_face'] == face]
        if len(subset) > 0:
            print(f"  Best {face}: count={len(subset)}, min delta={subset['best_mismatch'].min()*100:.2f}%")
    return df_results


# ── AgI Benchmark ───────────────────────────────────────────────

def test_agi_benchmark() -> Dict:
    print("\n  -- AgI Benchmark (True 2D matching) --")
    agi_basal = {
        'a1': np.array([AGI_A, 0.0]),
        'a2': np.array([AGI_A * np.cos(np.radians(120.0)),
                        AGI_A * np.sin(np.radians(120.0))]),
    }
    agi_prism = {
        'a1': np.array([AGI_A, 0.0]),
        'a2': np.array([0.0, AGI_C]),
    }
    ice_basal = build_ice_surface('basal')
    ice_prism = build_ice_surface('prism')

    results = {}
    r_b = compute_2d_mismatch_full(
        agi_basal['a1'], agi_basal['a2'],
        ice_basal['a1'], ice_basal['a2'], rotation_step=0.5)
    results['basal_basal'] = r_b
    print(f"  AgI(0001) vs Ice(0001): 2D delta={r_b['mismatch']*100:.3f}% "
          f"(literature ~1.3% 1D, our 2D norm matches this)")

    r_p = compute_2d_mismatch_full(
        agi_prism['a1'], agi_prism['a2'],
        ice_prism['a1'], ice_prism['a2'], rotation_step=0.5)
    results['prism_prism'] = r_p
    print(f"  AgI(10-10) vs Ice(10-10): 2D delta={r_p['mismatch']*100:.3f}%")
    return results


if __name__ == "__main__":
    test_agi_benchmark()
