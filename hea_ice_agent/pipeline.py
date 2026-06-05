#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEA-IceAgent Pipeline v2 — True 2D lattice matching + provenance.
"""

import os
import json
import time
import numpy as np
import pandas as pd

from .config import DATA_RAW_DIR, RESULTS_DIR, N_JOBS
from .download import download_all, check_data_files
from .loader import load_and_merge, clean_dataset, get_composition_groups
from .features import compute_all_features
from .lattice_matching import run_lattice_matching, test_agi_benchmark
from .scoring import compute_all_scores, get_top_candidates, record_provenance


class HEAIcePipeline:
    """HEA Ice Nucleation Screening Pipeline (v2)."""

    def __init__(self, n_jobs=N_JOBS, verbose=True):
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.df_clean = None
        self.df_features = None
        self.df_lattice = None
        self.df_scored = None
        self.composition_groups = None
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(RESULTS_DIR, 'figures'), exist_ok=True)

    def run_full(self, skip_download=True) -> dict:
        t0 = time.time()

        print("\n" + "=" * 70)
        print("  HEA-IceAgent v2 — True 2D Matching + DFT Data + Provenance")
        print("=" * 70)

        if not skip_download:
            download_all()

        # Phase 1
        self.df_clean = clean_dataset(load_and_merge())
        self.composition_groups = get_composition_groups(self.df_clean)

        # AgI benchmark
        test_agi_benchmark()

        # Phase 2
        self.df_features = compute_all_features(self.df_clean)

        # Phase 3
        self.df_lattice = run_lattice_matching(self.composition_groups, n_jobs=self.n_jobs, verbose=self.verbose)

        # Phase 4
        df_merged = self.df_features.merge(self.df_lattice, on='chemical_system', how='left', suffixes=('', '_lat'))
        self.df_scored = compute_all_scores(df_merged)
        self.df_scored = self.df_scored.drop_duplicates(subset=['chemical_system'], keep='first')

        # Provenance
        prov = record_provenance(self.df_clean, self.df_lattice)
        prov['data_stats']['n_structures_clean'] = len(self.df_clean)
        with open(os.path.join(RESULTS_DIR, 'provenance.json'), 'w', encoding='utf-8') as f:
            json.dump(prov, f, indent=2, ensure_ascii=False, default=str)

        # Phase 5
        from .report import generate_all_reports
        outputs = generate_all_reports(self.df_scored, RESULTS_DIR)

        print(f"\n  Pipeline complete in {time.time()-t0:.0f}s")
        return outputs
