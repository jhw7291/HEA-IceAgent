#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Engineering for HEA-IceAgent.

Generates domain-specific descriptors for three tasks:
1. Ice nucleation promotion (like AgI)
2. Ice nucleation inhibition
3. Ice recrystallization inhibition (IRI)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from .config import (
    ELEMENTS,
    ATOMIC_RADII,
    ELECTRONEGATIVITY,
    EFFECTIVE_BCC,
    EFFECTIVE_FCC,
    N_JOBS,
)
from .utils import (
    compute_composition_from_formula,
    configurational_entropy,
    atomic_radius_mismatch,
    electronegativity_variance,
    weighted_average,
    min_max_normalize,
)


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all engineered features for the dataset.

    Args:
        df: Cleaned dataset from loader

    Returns:
        DataFrame with engineered features added
    """
    df = df.copy()

    print("\n" + "=" * 60)
    print("Phase 2: Feature Engineering")
    print("=" * 60)

    # ── Composition-based features ──────────────────────────────
    print("  Computing composition features ...")

    # Parse compositions
    compositions = {}
    for cs in df['chemical_system'].unique():
        compositions[cs] = compute_composition_from_formula(str(cs))

    df['_comp_dict'] = df['chemical_system'].map(compositions)

    # Configurational entropy
    df['S_conf'] = df['_comp_dict'].apply(configurational_entropy)
    print(f"    Config entropy range: [{df['S_conf'].min():.3f}, {df['S_conf'].max():.3f}]")

    # Atomic radius mismatch (delta_r)
    df['delta_r'] = df['_comp_dict'].apply(
        lambda c: atomic_radius_mismatch(c, ATOMIC_RADII)
    )
    print(f"    delta_r range: [{df['delta_r'].min():.3f}, {df['delta_r'].max():.3f}]")

    # Electronegativity variance
    df['var_EN'] = df['_comp_dict'].apply(
        lambda c: electronegativity_variance(c, ELECTRONEGATIVITY)
    )
    print(f"    var_EN range: [{df['var_EN'].min():.4f}, {df['var_EN'].max():.4f}]")

    # Weighted average electronegativity
    df['avg_EN'] = df['_comp_dict'].apply(
        lambda c: weighted_average(c, ELECTRONEGATIVITY)
    )
    print(f"    avg_EN range: [{df['avg_EN'].min():.3f}, {df['avg_EN'].max():.3f}]")

    # Composition complexity (number of elements)
    df['n_elements'] = df['_comp_dict'].apply(len)

    # ── SRO-based features ──────────────────────────────────────
    print("  Computing SRO features ...")
    sro_cols = [f'mean abs SRO{i}' for i in range(1, 5)]
    sro_available = [c for c in sro_cols if c in df.columns]

    if sro_available:
        # SRO gradient (difference between first and fourth shell)
        if 'mean abs SRO1' in df.columns and 'mean abs SRO4' in df.columns:
            df['SRO_gradient'] = (
                df['mean abs SRO1'] - df['mean abs SRO4']
            ).abs()
        else:
            df['SRO_gradient'] = 0.0

        # SRO heterogeneity: standard deviation across shells
        if len(sro_available) >= 2:
            df['SRO_heterogeneity'] = df[sro_available].std(axis=1)
        else:
            df['SRO_heterogeneity'] = 0.0

        # Mean SRO level
        df['SRO_mean'] = df[sro_available].mean(axis=1)

        # SRO range (max-min across shells)
        if len(sro_available) >= 2:
            df['SRO_range'] = df[sro_available].max(axis=1) - df[sro_available].min(axis=1)
        else:
            df['SRO_range'] = 0.0

        print(f"    SRO_gradient range: [{df['SRO_gradient'].min():.4f}, "
              f"{df['SRO_gradient'].max():.4f}]")
        print(f"    SRO_heterogeneity range: [{df['SRO_heterogeneity'].min():.4f}, "
              f"{df['SRO_heterogeneity'].max():.4f}]")
    else:
        print("    Warning: No SRO columns found, using defaults")
        df['SRO_gradient'] = 0.0
        df['SRO_heterogeneity'] = 0.0
        df['SRO_mean'] = 0.0
        df['SRO_range'] = 0.0

    # ── Stability features ──────────────────────────────────────
    print("  Computing stability features ...")

    if 'Ef_per_atom' in df.columns:
        # Stability score: negative formation energy = more stable
        df['stability_score'] = -df['Ef_per_atom'].clip(lower=-0.6, upper=0.6)

        # Absolute stability: how close to the convex hull
        # (formation energy near 0 means potentially unstable vs pure elements)
        df['abs_stability'] = np.abs(df['Ef_per_atom'])

    # ── Hexagonality proxy ─────────────────────────────────────
    print("  Computing hexagonality proxy ...")

    if 'space_group_number' in df.columns:
        # Hexagonal space groups: 168-194
        df['is_hexagonal'] = df['space_group_number'].between(168, 194).astype(float)

        # Distance to hexagonal: normalized by space group proximity
        df['hex_proximity'] = 1.0 - np.abs(df['space_group_number'] - 190).clip(upper=50) / 50.0
    else:
        df['is_hexagonal'] = 0.0
        df['hex_proximity'] = 0.0

    # ── Hydrophobicity proxy ───────────────────────────────────
    print("  Computing hydrophobicity proxy ...")
    # Higher avg_EN -> more hydrophilic (water adsorbs via H-bonds)
    # Lower avg_EN variance -> more uniform surface -> less pinning
    df['hydrophobicity'] = 1.0 - min_max_normalize(df['avg_EN'])
    # Combine with electronegativity variance:
    # high EN variance -> heterogeneous surface -> more ice nucleation sites
    df['surface_homogeneity'] = 1.0 - min_max_normalize(df['var_EN'])

    # ── Surface energy proxy ───────────────────────────────────
    print("  Computing surface energy proxy ...")
    # Based on formation energy: more negative Ef -> higher surface energy
    if 'Ef_per_atom' in df.columns:
        df['surface_energy_proxy'] = (
            1.0 - min_max_normalize(df['Ef_per_atom'].abs())
        )
    else:
        df['surface_energy_proxy'] = 0.5

    # ── Work function proxy ────────────────────────────────────
    print("  Computing work function proxy ...")
    # Based on electronegativity: work function ~ 2.7 + 0.6*EN (eV)
    df['work_function_proxy'] = 2.7 + 0.6 * df['avg_EN']

    # ── Mixing enthalpy proxy ──────────────────────────────────
    print("  Computing mixing enthalpy proxy ...")
    # Approximated from delta_r and formation energy
    # Large delta_r + negative Ef -> large negative mixing enthalpy
    df['mixing_enthalpy_proxy'] = -df['delta_r'] * (
        -df.get('Ef_per_atom', pd.Series(0, index=df.index)).clip(upper=0)
    )

    # ── Lattice distortion index ───────────────────────────────
    print("  Computing lattice distortion index ...")
    # Combined metric of delta_r and var_EN
    df['lattice_distortion'] = (
        min_max_normalize(df['delta_r']) * 0.6 +
        min_max_normalize(df['var_EN']) * 0.4
    )

    # ── Electronic smoothness ──────────────────────────────────
    print("  Computing electronic smoothness ...")
    # Low EN variance -> electronically smooth surface -> poor nucleation
    # High EN variance -> many different local electronic environments
    df['electronic_smoothness'] = 1.0 - min_max_normalize(df['var_EN'])

    # ── Clean up temporary columns ─────────────────────────────
    df = df.drop(columns=['_comp_dict'], errors='ignore')

    print(f"  Total features computed: {_count_engineered_features(df)} new columns")
    return df


def _count_engineered_features(df: pd.DataFrame) -> int:
    """Count how many engineered feature columns were added."""
    engineered = [
        'S_conf', 'delta_r', 'var_EN', 'avg_EN', 'n_elements',
        'SRO_gradient', 'SRO_heterogeneity', 'SRO_mean', 'SRO_range',
        'stability_score', 'abs_stability',
        'is_hexagonal', 'hex_proximity',
        'hydrophobicity', 'surface_homogeneity',
        'surface_energy_proxy', 'work_function_proxy',
        'mixing_enthalpy_proxy', 'lattice_distortion',
        'electronic_smoothness',
    ]
    return sum(1 for c in engineered if c in df.columns)


def get_promoter_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features relevant for ice nucleation promotion.

    These features help identify materials like AgI that promote ice formation.
    """
    cols = [
        'S_conf', 'delta_r', 'stability_score',
        'is_hexagonal', 'hex_proximity',
        'surface_energy_proxy', 'work_function_proxy',
        'SRO_mean', 'avg_EN',
    ]
    available = [c for c in cols if c in df.columns]
    return df[available].copy()


def get_inhibitor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features relevant for ice nucleation inhibition.

    These features help identify materials that prevent ice from forming.
    """
    cols = [
        'hydrophobicity', 'surface_homogeneity',
        'electronic_smoothness', 'avg_EN', 'var_EN',
        'SRO_mean', 'SRO_range',
        'stability_score',
    ]
    available = [c for c in cols if c in df.columns]
    return df[available].copy()


def get_iri_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features relevant for ice recrystallization inhibition.

    These features help identify materials that prevent existing ice
    crystals from growing larger.
    """
    cols = [
        'SRO_heterogeneity', 'SRO_gradient', 'SRO_range',
        'S_conf', 'delta_r', 'lattice_distortion',
        'mixing_enthalpy_proxy', 'var_EN',
        'n_elements',
    ]
    available = [c for c in cols if c in df.columns]
    return df[available].copy()


if __name__ == "__main__":
    from .loader import load_and_merge, clean_dataset
    df = load_and_merge()
    df = clean_dataset(df)
    df = compute_all_features(df)
    print(df.columns.tolist())
