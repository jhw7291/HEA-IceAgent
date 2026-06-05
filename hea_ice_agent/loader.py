#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loader for HEA-IceAgent.
Loads from the 4 local CSV files in the agent directory.
"""

import os
import numpy as np
import pandas as pd

from .config import (
    DATA_RAW_DIR,
    FILE_STRUCTURE_INI,
    FILE_STRUCTURE_FINAL,
    FILE_SRO,
    FILE_HEA_MAIN,
    ELEMENTS,
    EF_PER_ATOM_RANGE,
    MIN_NELEMENTS,
    MAX_NELEMENTS,
)


def load_and_merge() -> pd.DataFrame:
    """Load all datasets and merge into a unified DataFrame.

    Uses the local CSV files: hea.2023-04-06.csv (raw DFT data),
    structure_ini_featurized.dat_all.csv (273 Matminer features),
    structure_featurized.dat_all.csv, and SROs_structure_ini.csv.

    Returns:
        Merged DataFrame with all structural, energetic, and SRO data
    """
    print("\n" + "=" * 60)
    print("Phase 1: Data Loading & Merging")
    print("=" * 60)

    # Load featurized initial structures (has 273 Matminer features + Ef_per_atom, lattice, etc)
    fname = FILE_STRUCTURE_INI
    path = os.path.join(DATA_RAW_DIR, fname)
    print(f"  Loading {fname} ...")
    df = pd.read_csv(path, index_col=0)
    print(f"    Shape: {df.shape}")

    # Load SRO data and merge
    sro_path = os.path.join(DATA_RAW_DIR, FILE_SRO)
    print(f"  Loading {FILE_SRO} ...")
    df_sro = pd.read_csv(sro_path, index_col=0)
    sro_cols = [f'mean abs SRO{i}' for i in range(1, 5)]
    sro_present = [c for c in sro_cols if c in df_sro.columns]
    if sro_present:
        df = pd.concat([df, df_sro[sro_present]], axis=1)
        print(f"    Merged {len(sro_present)} SRO columns")

    print(f"\n  Final: {df.shape[0]} structures, {df.shape[1]} columns")

    # Report key columns
    key_cols = ['Ef_per_atom', 'lattice', 'space_group_number',
                'NIONS', 'nelements', 'chemical_system', 'reduced_formula']
    found = [c for c in key_cols if c in df.columns]
    missing = [c for c in key_cols if c not in df.columns]
    if found:
        print(f"  Key columns found: {found}")
    if missing:
        print(f"  Key columns NOT found: {missing}")

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and filter the dataset."""
    print("\n  Cleaning dataset ...")
    initial_count = len(df)

    if 'Ef_per_atom' in df.columns:
        df = df.dropna(subset=['Ef_per_atom'])

    if 'nelements' in df.columns:
        df = df[(df['nelements'] >= MIN_NELEMENTS) &
                (df['nelements'] <= MAX_NELEMENTS)]

    if 'Ef_per_atom' in df.columns:
        low, high = EF_PER_ATOM_RANGE
        df = df[(df['Ef_per_atom'] >= low) & (df['Ef_per_atom'] <= high)]

    df = df[~df.index.duplicated(keep='first')]

    n_removed = initial_count - len(df)
    print(f"  Removed {n_removed} rows ({n_removed/initial_count*100:.1f}%)")
    print(f"  Clean dataset: {len(df)} structures")

    print(f"\n  -- Data Summary --")
    if 'nelements' in df.columns:
        for n in sorted(df['nelements'].unique()):
            count = (df['nelements'] == n).sum()
            print(f"    {n}-component: {count}")

    if 'lattice' in df.columns:
        for lat in df['lattice'].unique():
            count = (df['lattice'] == lat).sum()
            print(f"    {lat}: {count}")

    if 'NIONS' in df.columns:
        print(f"    NIONS range: {df['NIONS'].min()} - {df['NIONS'].max()}")

    if 'Ef_per_atom' in df.columns:
        ef = df['Ef_per_atom']
        print(f"    Ef_per_atom: {ef.mean():.3f} +- {ef.std():.3f} eV/atom")

    return df


def get_composition_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Group structures by unique chemical_system."""
    groups = df.groupby('chemical_system').agg(
        lattice=('lattice', lambda x: x.mode().iloc[0] if len(x.mode()) > 0
                 else x.iloc[0]),
        nelements=('nelements', 'first'),
        n_structures=('chemical_system', 'count'),
        mean_Ef=('Ef_per_atom', 'mean'),
        std_Ef=('Ef_per_atom', 'std'),
        min_Ef=('Ef_per_atom', 'min'),
        max_Ef=('Ef_per_atom', 'max'),
    ).reset_index()

    print(f"\n  Unique chemical systems: {len(groups)}")
    print(f"    BCC: {(groups['lattice'] == 'bcc').sum()}")
    print(f"    FCC: {(groups['lattice'] == 'fcc').sum()}")

    return groups
