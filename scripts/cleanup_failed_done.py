#!/usr/bin/env python3
"""
Delete 'done' files for ligands NOT present in boltzina_results.csv.

Usage:
    python cleanup_failed_done.py <results_root_dir> [--dry-run]

Scans <results_root_dir>/<target>/<fold>/ directories.
For each fold:
  - Reads boltzina_results.csv to find ligand_idx values that were scored
  - Deletes out/<idx>/done for any idx NOT in the CSV (so re-run will process them)
  - Preserves done files for successfully scored ligands (so re-run skips them)
"""
import argparse
import csv
from pathlib import Path


def get_scored_indices(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()
    indices = set()
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                indices.add(int(row['ligand_idx']))
            except (KeyError, ValueError):
                pass
    return indices


def cleanup_fold(fold_dir: Path, dry_run: bool) -> tuple[int, int]:
    csv_path = fold_dir / "boltzina_results.csv"
    out_dir = fold_dir / "out"
    if not out_dir.exists():
        return 0, 0

    scored = get_scored_indices(csv_path)
    deleted = 0
    kept = 0

    for lig_dir in out_dir.iterdir():
        if not lig_dir.is_dir():
            continue
        try:
            idx = int(lig_dir.name)
        except ValueError:
            continue

        done_file = lig_dir / "done"
        if not done_file.exists():
            continue

        if idx in scored:
            kept += 1
        else:
            if not dry_run:
                done_file.unlink()
            deleted += 1

    return deleted, kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be deleted without actually deleting")
    args = parser.parse_args()

    total_deleted = 0
    total_kept = 0

    for target_dir in sorted(args.results_root.iterdir()):
        if not target_dir.is_dir():
            continue
        for fold_dir in sorted(target_dir.iterdir()):
            if not fold_dir.is_dir():
                continue
            deleted, kept = cleanup_fold(fold_dir, args.dry_run)
            csv_rows = kept + deleted  # approximation
            if deleted > 0 or kept > 0:
                mode = "[DRY RUN] " if args.dry_run else ""
                print(f"{mode}{target_dir.name}/{fold_dir.name}: "
                      f"kept {kept} done-files, {'would delete' if args.dry_run else 'deleted'} {deleted}")
            total_deleted += deleted
            total_kept += kept

    print(f"\nTotal: kept {total_kept}, {'would delete' if args.dry_run else 'deleted'} {total_deleted} done-files")


if __name__ == "__main__":
    main()
