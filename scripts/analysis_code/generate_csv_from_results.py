#!/usr/bin/env python3
"""
Aggregate grasp success results saved under a directory structure
`<results_dir>/<robot>/<method>/<object>/grasp_success.npy` into a pandas DataFrame
and save a summary CSV. Also prints simple pivot tables.

Usage:
  python scripts/analysis/aggregate_grasp_results.py \
      --results-dir /home/cudagl/data/RAS_results \
      --out-csv /home/cudagl/data/RAS_results/summary_grasp_results.csv

The script looks for `grasp_success.npy` files under three-level dirs
robot/method/object and computes success count, trials, and rate.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import sys


def find_grasp_files(results_dir: Path):
    """Yield tuples (robot, method, object, path_to_file) for found grasp_success.npy files."""
    results_dir = results_dir.expanduser().resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for robot_dir in sorted(results_dir.iterdir()):
        if not robot_dir.is_dir():
            continue
        robot = robot_dir.name
        for method_dir in sorted(robot_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name
            for object_dir in sorted(method_dir.iterdir()):
                if not object_dir.is_dir():
                    continue
                obj = object_dir.name
                # prefer directly located file, otherwise search recursively
                candidate = object_dir / 'grasp_success.npy'
                if candidate.exists():
                    yield robot, method, obj, candidate
                    continue
                # fallback: search deeper for any grasp_success.npy
                found = list(object_dir.rglob('grasp_success.npy'))
                if found:
                    yield robot, method, obj, found[0]


def aggregate(results_dir: Path):
    records = []
    for robot, method, obj, fpath in find_grasp_files(results_dir):
        try:
            arr = np.load(fpath, allow_pickle=True)
        except Exception as e:
            print(f"Warning: failed loading {fpath}: {e}", file=sys.stderr)
            continue

        # normalize to 1D array of booleans/ints
        try:
            arr = np.asarray(arr).ravel()
        except Exception:
            arr = np.array([arr])

        # handle empty
        n_trials = int(arr.size)
        if n_trials == 0:
            n_success = 0
            success_rate = float('nan')
        else:
            # ensure boolean/int
            try:
                n_success = int(np.sum(arr.astype(int)))
            except Exception:
                # fallback: count truthy
                n_success = int(np.sum([1 for x in arr if bool(x)]))
            success_rate = n_success / n_trials

        # collect num_correspondences.npy files under the object root (robot/method/object)
        object_root = results_dir.expanduser().resolve() / robot / method / obj
        corr_paths = list(object_root.rglob('num_correspondences.npy'))
        corr_values = []
        for cp in corr_paths:
            try:
                v = np.load(cp)
                # expect scalar int or 1-element array
                v = int(np.asarray(v).ravel()[0])
                corr_values.append(v)
            except Exception:
                # skip malformed
                continue

        if corr_values:
            corr_mean = float(np.mean(corr_values))
            corr_sum = int(np.sum(corr_values))
            corr_min = int(np.min(corr_values))
            corr_max = int(np.max(corr_values))
            corr_nfiles = int(len(corr_values))
        else:
            corr_mean = float('nan')
            corr_sum = 0
            corr_min = 0
            corr_max = 0
            corr_nfiles = 0

        records.append({
            'robot': robot,
            'method': method,
            'object': obj,
            'n_trials': n_trials,
            'n_success': n_success,
            'success_rate': success_rate,
            'file_path': str(fpath),
            'num_correspondences_mean': corr_mean,
            'num_correspondences_sum': corr_sum,
            'num_correspondences_min': corr_min,
            'num_correspondences_max': corr_max,
            'num_correspondences_nfiles': corr_nfiles
        })

    df = pd.DataFrame.from_records(records,
                                   columns=['robot', 'method', 'object', 'n_trials', 'n_success', 'success_rate', 'file_path',
                                            'num_correspondences_mean', 'num_correspondences_sum', 'num_correspondences_min',
                                            'num_correspondences_max', 'num_correspondences_nfiles'])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path('/home/cudagl/data/RAS_results'), help='Root results dir')
    parser.add_argument('--out-csv', type=Path, default=Path('/home/cudagl/data/RAS_results/summary_grasp_results.csv'), help='Output CSV path')
    parser.add_argument('--robot', type=str, default=None, help='If set, show pivot for this robot')
    args = parser.parse_args()

    df = aggregate(args.results_dir)

    if df.empty:
        print(f"No grasp_success.npy files found under {args.results_dir}")
        return

    # save CSV
    out_csv = args.out_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved summary CSV to: {out_csv}")

    # additionally, write per-object correspondence summary CSV
    try:
        obj_corr = df.groupby(['robot', 'object']).agg(
            num_correspondences_mean=('num_correspondences_mean', 'mean'),
            num_correspondences_sum=('num_correspondences_sum', 'sum'),
            num_correspondences_min=('num_correspondences_min', 'min'),
            num_correspondences_max=('num_correspondences_max', 'max'),
            num_correspondences_nfiles=('num_correspondences_nfiles', 'sum')
        ).reset_index()
        corr_out = out_csv.parent / 'object_correspondences.csv'
        obj_corr.to_csv(corr_out, index=False)
        print(f"Saved per-object correspondences CSV to: {corr_out}")
        # also write a mean-only CSV (robot,object,num_correspondences_mean)
        mean_only = obj_corr[['robot', 'object', 'num_correspondences_mean']].copy()
        mean_only_out = out_csv.parent / 'object_correspondences_mean.csv'
        mean_only.to_csv(mean_only_out, index=False)
        print(f"Saved per-object correspondences (mean only) CSV to: {mean_only_out}")
    except Exception as e:
        print(f"Warning: failed writing object correspondences CSV: {e}", file=sys.stderr)

    # print overall summary
    pd.set_option('display.width', 200)
    print('\nHead of aggregated table:')
    print(df.head(50).to_string(index=False))

    # show pivot: average success_rate per object x method for each robot (or overall)
    if args.robot:
        sub = df[df['robot'] == args.robot]
        if sub.empty:
            print(f"No records for robot '{args.robot}'")
        else:
            pivot = sub.pivot_table(index='object', columns='method', values='success_rate')
            print(f"\nPivot table (robot={args.robot}) - success_rate:")
            print(pivot.fillna('-').to_string())
    else:
        pivot = df.pivot_table(index='object', columns='method', values='success_rate', aggfunc='mean')
        print('\nPivot table (all robots averaged) - success_rate:')
        print(pivot.fillna('-').to_string())


if __name__ == '__main__':
    main()
