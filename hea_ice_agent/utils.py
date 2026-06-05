#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for HEA-IceAgent
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


def rotation_matrix_2d(theta_deg: float) -> np.ndarray:
    """2D rotation matrix for given angle in degrees."""
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def compute_lattice_constant_vegard(
    composition: Dict[str, float],
    lattice_type: str,
    element_lattice_map: dict,
) -> float:
    """Estimate HEA lattice constant via Vegard's law.

    Args:
        composition: {element: fraction} dict, fractions sum to 1
        lattice_type: 'bcc' or 'fcc'
        element_lattice_map: dict mapping element -> lattice constant

    Returns:
        Estimated lattice constant in Å
    """
    a_estimated = 0.0
    total_frac = 0.0
    for elem, frac in composition.items():
        if elem in element_lattice_map:
            a_estimated += frac * element_lattice_map[elem]
            total_frac += frac
    if total_frac > 0:
        a_estimated /= total_frac
    else:
        # fallback: average
        a_estimated = np.mean(list(element_lattice_map.values()))
    return a_estimated


def compute_composition_from_formula(reduced_formula: str) -> Dict[str, float]:
    """Parse a reduced formula string into element fractions.

    Handles formulas like 'Al2CrFeNi', 'CrFeCoNi', 'Al0.5CrFeNi1.5'.
    Uses simple regex-free parsing.

    Args:
        reduced_formula: reduced chemical formula string

    Returns:
        dict of {element: atomic_fraction}
    """
    comp = {}
    i = 0
    current_element = ""
    current_number = ""
    elements_list = [
        'Al', 'Si', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
        'Ti', 'V', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
        'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
        'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
        'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
        'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    ]

    s = reduced_formula.strip()
    while i < len(s):
        # Try to match a two-letter element
        if i + 1 < len(s) and s[i:i+2] in elements_list:
            elem = s[i:i+2]
            i += 2
        elif s[i] in elements_list or s[i].isalpha():
            elem = s[i]
            i += 1
        else:
            # skip non-element chars
            i += 1
            continue

        # read number after element
        num_str = ""
        while i < len(s) and (s[i].isdigit() or s[i] == '.'):
            num_str += s[i]
            i += 1

        if num_str:
            comp[elem] = comp.get(elem, 0.0) + float(num_str)
        else:
            comp[elem] = comp.get(elem, 0.0) + 1.0

    # Normalize to fractions
    total = sum(comp.values())
    if total > 0:
        return {k: v / total for k, v in comp.items()}

    return {}


def configurational_entropy(composition: Dict[str, float]) -> float:
    """Compute configurational entropy S_conf = -Σ c_i ln(c_i).

    Args:
        composition: {element: fraction}

    Returns:
        S_conf value (dimensionless, normalized per atom)
    """
    s_conf = 0.0
    for frac in composition.values():
        if frac > 1e-10:
            s_conf -= frac * np.log(frac)
    return s_conf


def atomic_radius_mismatch(composition: Dict[str, float], radii_map: dict) -> float:
    """Compute atomic radius mismatch δ_r for HEA phase stability.

    δ_r = sqrt(Σ c_i * (1 - r_i / r_avg)²)

    Args:
        composition: {element: fraction}
        radii_map: {element: atomic_radius}

    Returns:
        δ_r mismatch parameter
    """
    r_avg = sum(frac * radii_map.get(elem, 1.3)
                for elem, frac in composition.items())
    if r_avg < 1e-10:
        return 0.0

    delta_sq = 0.0
    for elem, frac in composition.items():
        r_i = radii_map.get(elem, r_avg)
        delta_sq += frac * (1.0 - r_i / r_avg) ** 2

    return np.sqrt(delta_sq)


def electronegativity_variance(composition: Dict[str, float], en_map: dict) -> float:
    """Compute weighted electronegativity variance for a composition.

    Var(χ) = Σ c_i * (χ_i - χ_avg)²

    Args:
        composition: {element: fraction}
        en_map: {element: electronegativity}

    Returns:
        Variance of electronegativity
    """
    en_avg = sum(frac * en_map.get(elem, 1.8)
                 for elem, frac in composition.items())
    var_en = 0.0
    for elem, frac in composition.items():
        en_i = en_map.get(elem, en_avg)
        var_en += frac * (en_i - en_avg) ** 2
    return var_en


def weighted_average(composition: Dict[str, float], prop_map: dict) -> float:
    """Compute composition-weighted average of a property.

    Args:
        composition: {element: fraction}
        prop_map: {element: property_value}

    Returns:
        Weighted average
    """
    total = 0.0
    total_frac = 0.0
    for elem, frac in composition.items():
        if elem in prop_map:
            total += frac * prop_map[elem]
            total_frac += frac
    if total_frac > 0:
        return total / total_frac
    return 0.0


def score_to_percentile(scores: pd.Series) -> pd.Series:
    """Convert raw scores to percentiles (0-100)."""
    return scores.rank(pct=True) * 100.0


def min_max_normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]."""
    mn, mx = series.min(), series.max()
    if mx - mn < 1e-12:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def z_score_normalize(series: pd.Series) -> pd.Series:
    """Z-score normalize a series."""
    mu, std = series.mean(), series.std()
    if std < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / std


def best_n_per_composition(
    df: pd.DataFrame,
    score_col: str,
    group_col: str = 'chemical_system',
    n: int = 3,
    ascending: bool = False,
) -> pd.DataFrame:
    """For each unique composition group, keep the top N structures by score.

    This reduces redundancy since a single composition has many structures
    (ordered, SQS with different sizes).
    """
    result = []
    for name, group in df.groupby(group_col):
        sorted_group = group.sort_values(score_col, ascending=ascending)
        result.append(sorted_group.head(n))
    return pd.concat(result, ignore_index=True)


def format_lattice_match_report(
    composition: str,
    lattice_type: str,
    a_estimated: float,
    best_ice_face: str,
    best_hea_surface: str,
    best_mismatch: float,
    best_angle: float,
    best_supercell: Tuple[int, int, int, int],
) -> str:
    """Format a lattice matching result for display."""
    return (
        f"{composition:<30s} {lattice_type:>4s} a={a_estimated:.3f}Å  "
        f"{best_hea_surface:>10s} ↔ {best_ice_face:<15s}  "
        f"δ={best_mismatch*100:.2f}%  θ={best_angle:.1f}°  "
        f"SC=({best_supercell[0]},{best_supercell[1]};{best_supercell[2]},{best_supercell[3]})"
    )
