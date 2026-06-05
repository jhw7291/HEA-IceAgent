#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEA-IceAgent — Main Entry Point

Usage:
    python main.py                     # Run full pipeline
    python main.py --task promoter     # Run with promoter focus
    python main.py --skip-download     # Skip data download
    python main.py --force-download    # Force re-download
    python main.py --n-jobs 4          # Limit parallel workers
    python main.py --top-n 20          # Top N candidates per task
    python main.py --benchmark         # Only run AgI benchmark test

Example:
    python main.py --skip-download --n-jobs 8 --top-n 30
"""

import sys
import os
import argparse

# Ensure the parent directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hea_ice_agent.config import (
    TOP_N_CANDIDATES,
    N_JOBS,
    RESULTS_DIR,
)
from hea_ice_agent.pipeline import HEAIcePipeline
from hea_ice_agent.lattice_matching import test_agi_benchmark
from hea_ice_agent.scoring import get_top_candidates


def main():
    parser = argparse.ArgumentParser(
        description='HEA-IceAgent: High-Entropy Alloy Ice Nucleation Screening',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Full pipeline
  python main.py --skip-download                    # Skip data download
  python main.py --n-jobs 8 --top-n 30             # Parallel, more candidates
  python main.py --benchmark                        # Only AgI validation
        """,
    )
    parser.add_argument('--task', type=str, default='all',
                       choices=['all', 'promoter', 'inhibitor', 'iri'],
                       help='Screen for a specific task (default: all)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip Phase 0 (data download)')
    parser.add_argument('--force-download', action='store_true',
                       help='Force re-download of all data')
    parser.add_argument('--n-jobs', type=int, default=N_JOBS,
                       help=f'Number of parallel workers (default: {N_JOBS})')
    parser.add_argument('--top-n', type=int, default=TOP_N_CANDIDATES,
                       help=f'Number of top candidates per task (default: {TOP_N_CANDIDATES})')
    parser.add_argument('--results-dir', type=str, default=RESULTS_DIR,
                       help=f'Output directory (default: {RESULTS_DIR})')
    parser.add_argument('--benchmark', action='store_true',
                       help='Only run the AgI benchmark test')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce verbosity')

    args = parser.parse_args()

    # Override top N
    import hea_ice_agent.config as cfg
    cfg.TOP_N_CANDIDATES = args.top_n

    # Benchmark only
    if args.benchmark:
        print("HEA-IceAgent: AgI Benchmark Validation")
        print("=" * 50)
        results = test_agi_benchmark()
        for key, val in results.items():
            mismatch_pct = val['mismatch'] * 100
            status = "PASS" if mismatch_pct < 3.0 else "WARN"
            print(f"  [{status}] {key}: mismatch={mismatch_pct:.3f}% "
                  f"(expected AgI/Ice basal ~1.3%)")
        return

    # Full pipeline
    pipeline = HEAIcePipeline(
        results_dir=args.results_dir,
        n_jobs=args.n_jobs,
        verbose=not args.quiet,
    )

    if args.task == 'all':
        outputs = pipeline.run_full(
            skip_download=args.skip_download,
            force_download=args.force_download,
        )

        if outputs:
            print("\nOutput files:")
            for name, path in sorted(outputs.items()):
                if os.path.exists(path) and os.path.isfile(path):
                    size_kb = os.path.getsize(path) / 1024
                    print(f"  {name}: {path} ({size_kb:.0f} KB)")
                elif os.path.isdir(path):
                    print(f"  {name}: {path}/")
    else:
        # Single task
        pipeline.run_full(
            skip_download=args.skip_download,
            force_download=args.force_download,
        )
        top = pipeline.run_single_task(args.task)
        print(f"\nTop {args.top_n} {args.task} candidates:")
        print(top.to_string(max_rows=min(args.top_n, 20)))


if __name__ == '__main__':
    main()
