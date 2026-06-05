#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data verification for HEA-IceAgent.

The DFT dataset must be downloaded manually from Zenodo:
  https://zenodo.org/records/10854500

Place these 4 CSV files directly in the hea_ice_agent/ directory:
  - structure_ini_featurized.dat_all.csv  (~390 MB)
  - structure_featurized.dat_all.csv      (~394 MB)
  - hea.2023-04-06.csv                    (~453 MB)
  - SROs_structure_ini.csv                (~43 MB)
"""

import os
from .config import (
    DATA_RAW_DIR,
    FILE_STRUCTURE_INI,
    FILE_STRUCTURE_FINAL,
    FILE_SRO,
    FILE_HEA_MAIN,
)


def download_all(force: bool = False) -> dict:
    """Verify all required data files are present.

    Returns:
        dict mapping filename -> local path for found files
    """
    print("\n" + "=" * 60)
    print("Phase 0: Data Verification")
    print("=" * 60)
    print("  Data source: Zenodo 10.5281/zenodo.10854500")
    print("  Download: https://zenodo.org/records/10854500")

    file_map = {
        FILE_STRUCTURE_INI: "Featurized initial structures (273 Matminer features)",
        FILE_STRUCTURE_FINAL: "Featurized relaxed structures",
        FILE_SRO: "Short-range order parameters (SRO1-SRO4)",
        FILE_HEA_MAIN: "Raw DFT data (formation energy, volume, lattice constant)",
    }

    results = {}
    for fname, desc in file_map.items():
        path = os.path.join(DATA_RAW_DIR, fname)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            print(f"  OK  {fname} ({size_mb:.0f} MB)")
            results[fname] = path
        else:
            print(f"  --  {fname} - NOT FOUND")
            print(f"      Download from https://zenodo.org/records/10854500")

    return results


def check_data_files() -> bool:
    """Return True if all required CSV files are present."""
    required = [FILE_STRUCTURE_INI, FILE_STRUCTURE_FINAL, FILE_SRO, FILE_HEA_MAIN]
    return all(os.path.exists(os.path.join(DATA_RAW_DIR, f)) for f in required)
