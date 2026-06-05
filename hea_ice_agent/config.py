#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEA-IceAgent Global Configuration
"""

import os
import numpy as np

# ── Paths ───────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = ROOT_DIR
DATA_RAW_DIR = AGENT_DIR  # Data files are in the agent directory itself
RESULTS_DIR = os.path.join(AGENT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# ── Data Source ──────────────────────────────────────────────────
# The DFT dataset is published at Zenodo (DOI: 10.5281/zenodo.10854500).
# Download instructions: see README.md or run loader.py with the CSV files
# already placed in the hea_ice_agent directory.
ZENODO_DOI = "10.5281/zenodo.10854500"
ZENODO_RECORD_URL = "https://zenodo.org/records/10854500"

# Local filenames
FILE_STRUCTURE_INI = "structure_ini_featurized.dat_all.csv"
FILE_STRUCTURE_FINAL = "structure_featurized.dat_all.csv"
FILE_SRO = "SROs_structure_ini.csv"
FILE_HEA_MAIN = "hea.2023-04-06.csv"

# ── HEA Element Set ──────────────────────────────────────────────
ELEMENTS = ['Al', 'Si', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu']
N_ELEMENTS = len(ELEMENTS)

# Element properties (for Vegard's law calculation)
# Atomic radii (Å) - metallic radii
ATOMIC_RADII = {
    'Al': 1.43, 'Si': 1.17, 'Cr': 1.28, 'Mn': 1.27,
    'Fe': 1.26, 'Co': 1.25, 'Ni': 1.25, 'Cu': 1.28,
}

# Pauling electronegativity
ELECTRONEGATIVITY = {
    'Al': 1.61, 'Si': 1.90, 'Cr': 1.66, 'Mn': 1.55,
    'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90,
}

# BCC lattice constants (Å) - pure elements
BCC_LATTICE = {
    'Cr': 2.88, 'Mn': 2.89, 'Fe': 2.87,
}

# FCC lattice constants (Å) - pure elements
FCC_LATTICE = {
    'Al': 4.05, 'Ni': 3.52, 'Cu': 3.61,
}

# Si is diamond cubic, but in HEA context use an effective metallic radius
# For Si in BCC/FCC HEA, we approximate using a Vegard-equivalent value
# Si effective BCC: ~2.97, effective FCC: ~3.85 (estimated from DFT data)
EFFECTIVE_BCC = {
    'Al': 3.24, 'Si': 2.97, 'Cr': 2.88, 'Mn': 2.89,
    'Fe': 2.87, 'Co': 2.82, 'Ni': 2.80, 'Cu': 2.86,
}
EFFECTIVE_FCC = {
    'Al': 4.05, 'Si': 3.85, 'Cr': 3.62, 'Mn': 3.64,
    'Fe': 3.59, 'Co': 3.53, 'Ni': 3.52, 'Cu': 3.61,
}

# ── Ice Ih Crystal Constants ─────────────────────────────────────
ICE_A = 4.52   # Å (basal plane lattice constant)
ICE_C = 7.36   # Å (c-axis lattice constant)
ICE_GAMMA = 120.0  # degrees

# AgI reference (hexagonal, β-AgI at low temp)
AGI_A = 4.58  # Å
AGI_C = 7.49  # Å
# Known mismatch AgI(0001) vs Ice(0001): ~1.3%

# ── Ice Ih Surface Definitions ───────────────────────────────────
# 2D unit cell basis vectors for each ice face
# (a1_x, a1_y), (a2_x, a2_y)  in Å
ICE_SURFACES = {
    'basal': {
        'name': 'Basal (0001)',
        'miller': (0, 0, 0, 1),
        'a1': np.array([ICE_A, 0.0]),
        'a2': np.array([ICE_A * np.cos(np.radians(ICE_GAMMA)),
                        ICE_A * np.sin(np.radians(ICE_GAMMA))]),
    },
    'prism': {
        'name': 'Prism (10-10)',
        'miller': (1, 0, -1, 0),
        'a1': np.array([ICE_A, 0.0]),
        'a2': np.array([0.0, ICE_C]),
    },
    'pyramidal': {
        'name': 'Pyramidal (11-22)',
        'miller': (1, 1, -2, 2),
        'a1': np.array([ICE_A, 0.0]),
        # (11-22): the second vector is at ~58° to a1
        'a2': np.array([-ICE_A / 2,
                        np.sqrt(ICE_C**2 + (ICE_A * np.sqrt(3) / 2)**2)]),
    },
}

# ── HEA Surface Definitions ─────────────────────────────────────
# BCC surfaces: (110), (100), (111)
# FCC surfaces: (111), (100), (110)
# Each defined as 2D basis vectors (a1, a2) scaling with a_lattice

def bcc_surfaces(a):
    """Return 2D unit cells for BCC low-index surfaces given lattice constant a."""
    return {
        'BCC(110)': {
            'name': 'BCC (110)',
            'a1': np.array([np.sqrt(2) * a / 2, 0.0]),
            'a2': np.array([0.0, a]),
        },
        'BCC(100)': {
            'name': 'BCC (100)',
            'a1': np.array([a, 0.0]),
            'a2': np.array([0.0, a]),
        },
        'BCC(111)': {
            'name': 'BCC (111)',
            'a1': np.array([np.sqrt(2) * a / 2, 0.0]),
            'a2': np.array([np.sqrt(2) * a / 4,
                            np.sqrt(6) * a / 4]),
        },
    }

def fcc_surfaces(a):
    """Return 2D unit cells for FCC low-index surfaces given lattice constant a."""
    return {
        'FCC(111)': {
            'name': 'FCC (111)',
            'a1': np.array([np.sqrt(2) * a / 2, 0.0]),
            'a2': np.array([np.sqrt(2) * a / 4,
                            np.sqrt(6) * a / 4]),
        },
        'FCC(100)': {
            'name': 'FCC (100)',
            'a1': np.array([a, 0.0]),
            'a2': np.array([0.0, a]),
        },
        'FCC(110)': {
            'name': 'FCC (110)',
            'a1': np.array([np.sqrt(2) * a / 2, 0.0]),
            'a2': np.array([0.0, a]),
        },
    }

# ── Lattice Matching Parameters ─────────────────────────────────
# Maximum supercell multiplier to search
MAX_SUPERCELL = (4, 4, 4, 4)  # (i_max, j_max, k_max, l_max)
# Rotation angle step (degrees) for searching optimal lattice match
ROTATION_STEP = 1.0  # 1° resolution
MAX_AREA_RATIO = 2.0  # max allowed area ratio between substrate and ice supercells
# Number of top matches to track per HEA composition
N_TOP_MATCHES = 3

# ── Scoring Weights ─────────────────────────────────────────────
# Task A: Ice Nucleation Promoter Score (like AgI)
PROMOTER_WEIGHTS = {
    'lattice_match': 0.45,   # exp(-δ²/2σ²), σ=5%
    'stability': 0.20,       # normalized formation energy
    'hexagonality': 0.15,    # proximity to hexagonal structure
    'surface_energy_proxy': 0.10,
    'work_function_proxy': 0.10,
}
PROMOTER_MISMATCH_SIGMA = 0.05  # 5% mismatch tolerance (AgI~1.3%)

# Task B: Ice Nucleation Inhibitor Score
INHIBITOR_WEIGHTS = {
    'lattice_mismatch': 0.40,  # 1 - exp(-δ²/2σ²), higher mismatch better
    'hydrophobicity': 0.25,    # based on electronegativity + surface energy
    'electronic_smoothness': 0.20,  # low EN variance
    'surface_smoothness': 0.15,     # low SRO
}

# Task C: Ice Recrystallization Inhibitor (IRI) Score
IRI_WEIGHTS = {
    'sro_heterogeneity': 0.30,    # SRO gradient/variance
    'config_entropy': 0.25,       # high configurational entropy
    'lattice_distortion': 0.25,   # atomic radius mismatch
    'mixing_enthalpy_proxy': 0.20, # based on formation energy
}

# ── Filtering Parameters ─────────────────────────────────────────
MIN_NELEMENTS = 3       # at least ternary
MAX_NELEMENTS = 7
EF_PER_ATOM_RANGE = (-0.6, 0.6)  # eV/atom, keep reasonably stable compositions

# ── Output ──────────────────────────────────────────────────────
TOP_N_CANDIDATES = 50  # Number of top candidates per task
OUTPUT_FORMATS = ['csv', 'html', 'json']

# ── CPU Settings ─────────────────────────────────────────────────
N_JOBS = -1  # Use all available cores
XGB_TREE_METHOD = "hist"  # CPU-only XGBoost
