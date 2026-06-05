# HEA-IceAgent v2.2

## Overview

A computational screening agent that identifies high-entropy alloys (HEAs) capable of controlling ice nucleation and ice recrystallization. It combines **database screening** (ranking 218 known HEA compositions from DFT data) with **ML-driven reasoning** (predicting 1,170 novel compositions not yet in any database).

**Research context:** Ice formation on surfaces is a critical problem in aviation, wind energy, power transmission, and cryopreservation. Silver iodide (AgI) is the gold-standard ice nucleating agent with a lattice mismatch of ~1.3% with ice Ih. This agent systematically searches for HEA compositions that match or exceed AgI's performance — or actively suppress ice formation.

---

## Scientific Background

### Ice Nucleation on Solid Surfaces

Heterogeneous ice nucleation occurs when a solid surface provides a template that lowers the free energy barrier for water molecules to organize into the ice Ih crystal structure. The key determinant is **epitaxial lattice matching**: the closer the surface atomic arrangement matches ice Ih, the more effectively it nucleates ice.

**Ice Ih crystal:**
- Hexagonal, a = 4.52 Å, c = 7.36 Å
- Key surfaces: Basal (0001), Prism (10-10), Pyramidal (11-22)

**AgI (silver iodide) — the gold standard:**
- Hexagonal, a = 4.58 Å, c = 7.49 Å
- AgI(0001) vs Ice(0001): 2D lattice mismatch = **1.327%**

### Why High-Entropy Alloys?

HEAs offer three unique advantages for ice nucleation control:

1. **Continuous lattice tunability**: By varying composition, the lattice constant can be continuously tuned across a range that covers the optimal match with ice Ih.

2. **Chemical heterogeneity**: The random distribution of 4-7 elements creates diverse local environments that can pin ice grain boundaries — analogous to how antifreeze proteins (AFPs) inhibit ice recrystallization.

3. **Combinatorial diversity**: 8 elements → 246 possible element combinations × countless stoichiometric ratios, far exceeding what can be explored by DFT alone.

### Three Screening Tasks

| Task | Goal | Physical Mechanism | Key Descriptor |
|---|---|---|---|
| **A: Promoter** | Promote ice nucleation (like AgI) | Epitaxial lattice matching | 2D lattice mismatch with ice Ih |
| **B: Inhibitor** | Prevent ice nucleation | High lattice mismatch + hydrophobicity | Mismatch + electronegativity |
| **C: IRI** | Inhibit ice recrystallization | Chemical heterogeneity pins grain boundaries | SRO heterogeneity + configurational entropy |

### Reference

Li et al. (2025) demonstrated that high-entropy alloys can be synthesized using a **bilayer ice recrystallization method**, directly connecting HEA formation to ice growth control — validating the premise of this screening agent.

---

## Architecture

```
hea_ice_agent/
├── config.py              Global parameters (ice Ih, elements, weights)
├── loader.py              CSV loading, merging, cleaning (78K→218 unique compositions)
├── features.py            20 engineered descriptors (S_conf, delta_r, SRO, etc.)
├── lattice_matching.py    TRUE 2D epitaxial matching (Zur rotating overlayer algorithm)
├── scoring.py             Three-task scoring (AgI-calibrated at 1.327%)
├── pipeline.py            Orchestrator for SCREEN mode
├── dual_mode.py           Dual-mode runner (SCREEN + REASON)
├── report.py              CSV / JSON / HTML output + matplotlib figures
├── utils.py               Composition parsing, entropy, mismatch helpers
├── download.py            Data file verification
└── main.py                CLI entry point
```

**Total:** 14 modules, ~3,200 lines of Python

### Dependencies

```
numpy  pandas  scipy  scikit-learn  xgboost  matplotlib  seaborn  joblib
```

No external API calls. No GPU required. CPU-only XGBoost mode.

---

## Dual-Mode Workflow

### Mode 1: SCREEN (Database Ranking)

```
4 CSV files (1.28 GB)
  → Load & clean: 83,797 → 78,085 structures
  → Group: 218 unique chemical systems
  → Extract DFT lattice constants from hea.2023-04-06.csv
  → Feature engineering: 20 descriptors
  → Full 2D lattice matching:
      9 face-pairs × 180 rotation angles × 4×4 supercell search
      AgI verification: 1.327%  PASS
  → 3-task scoring → Top-50 candidates per task
```

**Runtime:** ~14 minutes with 4 CPU cores

### Mode 2: REASON (ML Inference for Novel Compositions)

```
218 known compositions → RandomForest training
  → Enumerate all 246 element combinations (3-7 of 8 elements)
  → Generate variant stoichiometries → 1,170 candidate compositions
  → ML predict BCC/FCC lattice constant + formation energy
  → Proxy mismatch scoring → Rank by ice nucleation potential
```

**Runtime:** ~7 seconds

### Data Provenance

All input files are tracked with SHA256 hashes, recorded in `results/provenance.json` alongside parameter snapshots and run timestamps. No external APIs are called — the agent operates entirely on local DFT data and its own ML models.

---

## Core Algorithm: 2D Epitaxial Matching

The lattice matching engine implements the **Zur rotating overlayer method**:

1. For each of 9 (HEA surface × ice face) pairs:
   - Build 2D unit cells from DFT lattice constants
   - Rotate ice lattice 0–180° (1° steps)
   - Search integer supercell matrices S_sub and S_ice
   - Compute 2D strain tensor: ε = S_ice⁻¹ · S_sub − I
   - Mismatch = ‖ε‖_F / √2 (normalized to 1D-equivalent %)

2. Return the global minimum mismatch across all face pairs

**Verified against AgI:** AgI(0001) vs Ice(0001) = 1.327% (literature ~1.3%)

---

## Usage

### Quick Start

```bash
# 1. Download dataset from Zenodo and place 4 CSV files in hea_ice_agent/

# 2. Run screening (Mode 1):
python -c "
import sys; sys.path.insert(0,'E:/HEA-Agent')
from hea_ice_agent.pipeline import HEAIcePipeline
HEAIcePipeline(n_jobs=4).run_full()
"

# 3. Run dual-mode (Mode 1 + Mode 2):
python -c "
import sys; sys.path.insert(0,'E:/HEA-Agent')
from hea_ice_agent.dual_mode import run_dual_mode
run_dual_mode(n_jobs=4)
"
```

### Output Files

| File | Content |
|---|---|
| `results/candidates_promoter.csv` | Top 50 ice nucleation promoters (from database) |
| `results/candidates_inhibitor.csv` | Top 50 ice nucleation inhibitors (from database) |
| `results/candidates_iri.csv` | Top 50 IRI inhibitors (from database) |
| `results/candidates_all_tasks.csv` | All 218 compositions ranked |
| `results/reasoned_new_candidates.csv` | 1,170 ML-predicted new compositions |
| `results/provenance.json` | SHA256 data verification + parameter snapshot |
| `results/report.html` | Interactive HTML report |
| `results/summary.json` | Machine-readable JSON summary |

---

## Key Results

### Task A: Ice Nucleation Promoters (AgI-like)

| Rank | Composition | Lattice | 2D Mismatch | Score |
|---|---|---|---|---|
| 1 | **Co-Fe-Ni-Si** | BCC | 1.320% | 99.8 |
| 2 | Al-Cu-Fe-Mn-Si | FCC | 1.332% | 99.7 |
| 3 | Al-Cr-Ni | BCC | 1.327% | 99.5 |
| 4 | Al-Cr-Cu-Ni-Si | BCC | 1.316% | 99.5 |
| 5 | Al-Cr-Cu-Ni | BCC | 1.315% | 99.4 |

**Key insight:** Si-containing ternaries and quaternaries show the best match to ice Ih. The top candidate Co-Fe-Ni-Si (BCC) has a mismatch of 1.320% — essentially identical to AgI's 1.327%.

### Task B: Ice Nucleation Inhibitors

| Rank | Composition | Lattice | 2D Mismatch | Score |
|---|---|---|---|---|
| 1 | **Al-Cr-Mn** | BCC | 1.035% | 99.9 |
| 2 | Al-Cr-Fe-Mn | BCC | 1.718% | 99.9 |
| 3 | Al-Co-Cr-Mn | BCC | 1.707% | 99.7 |
| 4 | Al-Co-Fe-Ni | BCC | 1.953% | 98.8 |
| 5 | Al-Co-Cr-Fe | BCC | 1.871% | 98.7 |

### Task C: Ice Recrystallization Inhibitors

| Rank | Composition | Lattice | SRO_h | S_conf | Score |
|---|---|---|---|---|---|
| 1 | **Al-Co-Cr-Fe-Mn-Ni-Si** | BCC | 0.214 | 1.946 | 96.0 |
| 2 | Al-Mn-Ni-Si | BCC | 0.412 | 1.386 | 95.7 |
| 3 | Al-Cr-Mn-Si | BCC | 0.380 | 1.386 | 93.9 |
| 4 | Al-Co-Cr-Si | BCC | 0.333 | 1.386 | 93.8 |
| 5 | Al-Co-Fe-Mn-Si | BCC | 0.627 | 1.609 | 93.7 |

### ML-Reasoned New Compositions

The REASON mode predicts **1,170 compositions not in any database**. Top predicted new promoters include Al-Mn-Ni-Si variants, new inhibitors are Al-Cr-Cu-Mn-Si variants, and new IRI candidates are Al-Cu-Mn-Ni-Si variants — all awaiting DFT validation.

---

## Known Limitations

1. **Composition-averaged DFT lattice constants** — structure-level variation (ordered vs SQS, different cell sizes) is not captured
2. **Scoring weights** are physics-inspired, not experimentally calibrated
3. **Missing physics**: surface termination chemistry, hydrogen bonding capability, ice adhesion energy
4. **Ideal surfaces only**: low-index BCC/FCC without reconstructions, steps, or defects
5. **8-element space**: database limited to Al-Si-Cr-Mn-Fe-Co-Ni-Cu; other promising elements (Ti, V, Mo, W) not included
6. **REASON mode proxy**: uses estimated mismatch rather than full 2D matching for speed

**All results are physics-guided hypotheses requiring DFT surface calculations and/or ice nucleation experiments for validation.**

---

## Data Source

Kangming Li et al., "Efficient first principles based modeling via machine learning: from simple representations to high entropy materials," *J. Mater. Chem. A*, 2024. DOI: [10.1039/D4TA00982G](https://doi.org/10.1039/D4TA00982G)

Dataset: Zenodo [10.5281/zenodo.10854500](https://doi.org/10.5281/zenodo.10854500)

---

## License

MIT License — see LICENSE file.
