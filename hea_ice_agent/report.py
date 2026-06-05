#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report generation for HEA-IceAgent.

Generates:
- Top-N candidate CSV files for each task
- JSON summary with all rankings
- HTML comprehensive report
- Key visualization figures
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime

from .config import (
    RESULTS_DIR,
    FIGURES_DIR,
    TOP_N_CANDIDATES,
    ELEMENTS,
)
from .scoring import get_top_candidates


def generate_all_reports(
    df_scored: pd.DataFrame,
    results_dir: str = RESULTS_DIR,
) -> dict:
    """Generate all output reports.

    Args:
        df_scored: DataFrame with all scores
        results_dir: Output directory

    Returns:
        dict: {filename: filepath} for all generated files
    """
    print("\n" + "=" * 60)
    print("Phase 5: Report Generation")
    print("=" * 60)

    outputs = {}
    os.makedirs(results_dir, exist_ok=True)
    figs_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figs_dir, exist_ok=True)

    # ── CSV outputs ──
    print("\n  Generating CSV candidate lists ...")
    for task in ['promoter', 'inhibitor', 'iri']:
        top = get_top_candidates(df_scored, task=task, top_n=TOP_N_CANDIDATES)
        fpath = os.path.join(results_dir, f'candidates_{task}.csv')
        top.to_csv(fpath, index=False)
        outputs[f'candidates_{task}.csv'] = fpath
        print(f"    {fpath} ({len(top)} candidates)")

    # Combined ranking
    combined = _build_combined_ranking(df_scored)
    fpath = os.path.join(results_dir, 'candidates_all_tasks.csv')
    combined.to_csv(fpath, index=False)
    outputs['candidates_all_tasks.csv'] = fpath
    print(f"    {fpath} ({len(combined)} entries)")

    # ── JSON summary ──
    print("\n  Generating JSON summary ...")
    summary = _build_json_summary(df_scored)
    fpath = os.path.join(results_dir, 'summary.json')
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    outputs['summary.json'] = fpath
    print(f"    {fpath}")

    # ── HTML report ──
    print("\n  Generating HTML report ...")
    html = _build_html_report(df_scored, summary)
    fpath = os.path.join(results_dir, 'report.html')
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(html)
    outputs['report.html'] = fpath
    print(f"    {fpath}")

    # ── Figures ──
    print("\n  Generating figures ...")
    try:
        _generate_figures(df_scored, figs_dir)
        outputs['figures/'] = figs_dir
    except Exception as e:
        print(f"    Warning: Figure generation failed: {e}")

    return outputs


def _build_combined_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Build a combined ranking DataFrame with all three task scores."""
    cols = [
        'chemical_system', 'lattice_type', 'a_estimated',
        'best_mismatch', 'best_ice_face', 'best_hea_surface',
        'nelements', 'mean_Ef', 'n_structures',
        'S_conf', 'delta_r', 'SRO_heterogeneity',
        'S_promoter', 'S_promoter_norm',
        'S_inhibitor', 'S_inhibitor_norm',
        'S_iri', 'S_iri_norm',
    ]
    available = [c for c in cols if c in df.columns]
    combined = df[available].copy()

    # Round numeric columns
    for c in combined.select_dtypes(include=[np.number]).columns:
        combined[c] = combined[c].round(4)

    return combined.sort_values('S_promoter', ascending=False)


def _build_json_summary(df: pd.DataFrame) -> dict:
    """Build a JSON summary with statistics and top candidates."""
    summary = {
        'generated_at': datetime.now().isoformat(),
        'pipeline': 'HEA-IceAgent v2.1',
        'dataset_stats': {
            'n_compositions': len(df),
            'n_structures': int(df['n_structures'].sum()) if 'n_structures' in df.columns else 0,
        },
        'tasks': {},
    }

    for task in ['promoter', 'inhibitor', 'iri']:
        score_col = f'S_{task}'
        if score_col not in df.columns:
            continue

        top = get_top_candidates(df, task=task, top_n=10)
        summary['tasks'][task] = {
            'score_mean': float(df[score_col].mean()),
            'score_std': float(df[score_col].std()),
            'score_max': float(df[score_col].max()),
            'top_10': [],
        }

        for _, row in top.iterrows():
            entry = {
                'composition': str(row.get('chemical_system', 'N/A')),
                'lattice_type': str(row.get('lattice_type', 'N/A')),
                'lattice_constant': float(row.get('a_dft', row.get('a_estimated', 0))),
                'mismatch_percent': float(row.get('best_mismatch', 0)),
                'ice_face': str(row.get('best_ice_face', 'N/A')),
                'hea_surface': str(row.get('best_hea_surface', 'N/A')),
                'score': float(row.get(score_col, 0)),
                'score_norm': float(row.get(f'{score_col}_norm', 0)),
            }
            summary['tasks'][task]['top_10'].append(entry)

    return summary


def _build_html_report(df: pd.DataFrame, summary: dict) -> str:
    """Generate a self-contained HTML report."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HEA-IceAgent Screening Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px;
         margin: 0 auto; padding: 20px; background: #f5f5f5; }}
  h1 {{ color: #1a5276; border-bottom: 3px solid #2980b9; padding-bottom: 10px; }}
  h2 {{ color: #2471a3; margin-top: 30px; }}
  h3 {{ color: #2c3e50; }}
  table {{ width: 100%; border-collapse: collapse; margin: 10px 0;
           background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th {{ background: #2980b9; color: white; padding: 10px 8px; text-align: left;
        font-size: 0.9em; }}
  td {{ padding: 8px; border-bottom: 1px solid #ddd; font-size: 0.9em; }}
  tr:hover {{ background: #eaf2f8; }}
  .header {{ background: linear-gradient(135deg, #1a5276, #2980b9);
             color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px; }}
  .stat-box {{ display: inline-block; background: white; padding: 15px 25px;
               margin: 10px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .stat-value {{ font-size: 1.5em; font-weight: bold; color: #2980b9; }}
  .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px;
            font-size: 0.8em; font-weight: bold; }}
  .badge-bcc {{ background: #d4e6f1; color: #1a5276; }}
  .badge-fcc {{ background: #fadbd8; color: #922b21; }}
  .badge-good {{ background: #d5f5e3; color: #1e8449; }}
  .badge-warn {{ background: #fcf3cf; color: #7d6608; }}
  .badge-bad {{ background: #fadbd8; color: #922b21; }}
</style>
</head>
<body>
<div class="header">
  <h1 style="margin:0; border:none; color:white;">HEA-IceAgent Screening Report</h1>
  <p style="margin:10px 0 0 0; opacity:0.9;">
    High-Entropy Alloy Screening for Ice Nucleation Control |
    Generated: {summary.get('generated_at', 'N/A')}
  </p>
</div>

<h2>Dataset Overview</h2>
<div style="text-align:center;">
  <div class="stat-box">
    <div class="stat-value">{summary['dataset_stats']['n_compositions']:,}</div>
    <div>Unique Compositions</div>
  </div>
  <div class="stat-box">
    <div class="stat-value">8</div>
    <div>Element Types</div>
  </div>
  <div class="stat-box">
    <div class="stat-value">3</div>
    <div>Screening Tasks</div>
  </div>
  <div class="stat-box">
    <div class="stat-value">{TOP_N_CANDIDATES}</div>
    <div>Top Candidates/Task</div>
  </div>
</div>
"""

    # Task sections
    task_names = {
        'promoter': 'Task A: Ice Nucleation Promoters (AgI-like)',
        'inhibitor': 'Task B: Ice Nucleation Inhibitors',
        'iri': 'Task C: Ice Recrystallization Inhibitors (IRI)',
    }
    task_descriptions = {
        'promoter': 'Materials with low lattice mismatch to ice Ih, enabling epitaxial ice nucleation. High configurational entropy provides diverse nucleation sites.',
        'inhibitor': 'Materials with high lattice mismatch and hydrophobic character that suppress ice crystal formation on their surfaces.',
        'iri': 'Materials whose chemical heterogeneity and lattice distortion pin ice grain boundaries, preventing coarsening.',
    }

    for task in ['promoter', 'inhibitor', 'iri']:
        task_data = summary['tasks'].get(task, {})
        if not task_data:
            continue

        html += f"""
<h2>{task_names.get(task, task)}</h2>
<p>{task_descriptions.get(task, '')}</p>
<p>Score range: {task_data.get('score_mean', 0):.3f} ± {task_data.get('score_std', 0):.3f} (max: {task_data.get('score_max', 0):.3f})</p>

<h3>Top 10 Candidates</h3>
<table>
<thead>
<tr>
  <th>Rank</th>
  <th>Composition</th>
  <th>Lattice</th>
  <th>a (Å)</th>
  <th>Mismatch (%)</th>
  <th>Best Ice Face</th>
  <th>Best HEA Surface</th>
  <th>Score (norm)</th>
</tr>
</thead>
<tbody>"""

        for rank, entry in enumerate(task_data.get('top_10', []), 1):
            mismatch = entry.get('mismatch_percent', 0)
            if isinstance(mismatch, (int, float)):
                mismatch = f"{mismatch:.3f}"

            lattice = entry.get('lattice_type', 'N/A')
            badge_class = 'badge-bcc' if lattice == 'bcc' else 'badge-fcc'

            mis_val = float(mismatch) if mismatch != 'N/A' else 99
            mis_class = 'badge-good' if mis_val < 5 else 'badge-warn' if mis_val < 10 else 'badge-bad'

            html += f"""
<tr>
  <td><strong>#{rank}</strong></td>
  <td><strong>{entry.get('composition', 'N/A')}</strong></td>
  <td><span class="badge {badge_class}">{lattice}</span></td>
  <td>{entry.get('lattice_constant', 0):.3f}</td>
  <td><span class="badge {mis_class}">{mismatch}%</span></td>
  <td>{entry.get('ice_face', 'N/A')}</td>
  <td>{entry.get('hea_surface', 'N/A')}</td>
  <td>{entry.get('score_norm', 0):.1f}</td>
</tr>"""

        html += "\n</tbody></table>\n"

    # Footer
    html += f"""
<hr style="margin-top:40px;">
<p style="color:#888; font-size:0.85em;">
  HEA-IceAgent | Dataset: Zenodo 10.5281/zenodo.10854500 |
  Elements: {', '.join(ELEMENTS)} |
  Report generated {summary.get('generated_at', 'N/A')}
</p>
</body></html>"""

    return html


def _generate_figures(df: pd.DataFrame, figs_dir: str):
    """Generate publication-quality figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_palette("colorblind")
        sns.set_style("whitegrid")
    except ImportError:
        print("    matplotlib/seaborn not available, skipping figures")
        return

    # Figure 1: Mismatch distribution for each lattice type
    if 'best_mismatch' in df.columns and 'lattice_type' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        for lt in ['bcc', 'fcc']:
            data = df[df['lattice_type'] == lt]['best_mismatch'] * 100  # ratio -> percent
            if len(data) > 0:
                ax.hist(data, bins=50, alpha=0.5, label=f'{lt.upper()} (n={len(data)})')
        ax.axvline(x=1.327, color='green', linestyle='--', label='AgI (1.327%)')
        ax.set_xlabel('2D Lattice Mismatch with Ice Ih (%)')
        ax.set_ylabel('Number of Compositions')
        ax.set_title('HEA-Ice 2D Lattice Mismatch Distribution')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figs_dir, 'mismatch_distribution.png'), dpi=150)
        plt.close(fig)

    # Figure 2: Score comparison scatter
    score_cols = [c for c in ['S_promoter_norm', 'S_inhibitor_norm', 'S_iri_norm']
                  if c in df.columns]
    if len(score_cols) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        pairs = [(0, 1), (0, 2), (1, 2)]
        labels = ['Promoter', 'Inhibitor', 'IRI']
        for ax, (i, j) in zip(axes, pairs):
            if i < len(score_cols) and j < len(score_cols):
                ax.scatter(df[score_cols[i]], df[score_cols[j]],
                          alpha=0.3, s=10)
                ax.set_xlabel(f'{labels[i]} Score')
                ax.set_ylabel(f'{labels[j]} Score')
        fig.suptitle('Cross-Task Score Correlations')
        fig.tight_layout()
        fig.savefig(os.path.join(figs_dir, 'score_correlations.png'), dpi=150)
        plt.close(fig)

    # Figure 3: Top promoter candidates bar chart
    if 'S_promoter' in df.columns:
        top10 = get_top_candidates(df, task='promoter', top_n=20)
        if 'chemical_system' in top10.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.barh(range(len(top10)), top10['S_promoter'].values)
            ax.set_yticks(range(len(top10)))
            ax.set_yticklabels(top10['chemical_system'].values, fontsize=9)
            ax.set_xlabel('Promoter Score')
            ax.set_title('Top 20 Ice Nucleation Promoter Candidates')
            ax.invert_yaxis()
            # Color by lattice type
            if 'lattice_type' in top10.columns:
                for i, (_, row) in enumerate(top10.iterrows()):
                    bars[i].set_color('#2980b9' if row['lattice_type'] == 'bcc'
                                    else '#e74c3c')
            fig.tight_layout()
            fig.savefig(os.path.join(figs_dir, 'top_promoters.png'), dpi=150)
            plt.close(fig)

    print(f"    Figures saved to {figs_dir}/")
