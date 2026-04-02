"""
Judge Validation Aggregator

Reads judge_validate CSV exports from a folder and computes
aggregated Micro P/R/F1 for Baseline and Judge, plus Agreement %.

Usage:
    python aggregate_judge_validation.py <csv_folder> [output_file]

Example:
    python aggregate_judge_validation.py ./validation_results ./aggregated_results.csv

CSV input format (per file):
    judge,note,baseline_tp,baseline_fp,baseline_fn,judge_tp,judge_fp,judge_fn

Output format:
    judge,baseline_p,baseline_r,baseline_f1,judge_p,judge_r,judge_f1,agreement_pct
"""

import csv
import os
import sys
import glob


def read_csvs_from_folder(folder_path):
    """Read all CSV files from a folder and combine rows."""
    all_rows = []
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        sys.exit(1)

    for filepath in csv_files:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)
        print(f"Read {filepath}")

    print(f"Total rows loaded: {len(all_rows)}")
    return all_rows


def compute_metrics(rows):
    """Compute aggregated metrics per judge."""
    # Group rows by judge
    judges = {}
    for row in rows:
        judge = row["judge"]
        if judge not in judges:
            judges[judge] = []
        judges[judge].append(row)

    results = []
    for judge, judge_rows in judges.items():
        # Sum counts across all notes
        b_tp = sum(int(r["baseline_tp"]) for r in judge_rows)
        b_fp = sum(int(r["baseline_fp"]) for r in judge_rows)
        b_fn = sum(int(r["baseline_fn"]) for r in judge_rows)

        j_tp = sum(int(r["judge_tp"]) for r in judge_rows)
        j_fp = sum(int(r["judge_fp"]) for r in judge_rows)
        j_fn = sum(int(r["judge_fn"]) for r in judge_rows)

        # Micro P/R/F1 for Baseline
        baseline_p = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
        baseline_r = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
        baseline_f1 = (
            2 * baseline_p * baseline_r / (baseline_p + baseline_r)
            if (baseline_p + baseline_r) > 0
            else 0.0
        )

        # Micro P/R/F1 for Judge
        judge_p = j_tp / (j_tp + j_fp) if (j_tp + j_fp) > 0 else 0.0
        judge_r = j_tp / (j_tp + j_fn) if (j_tp + j_fn) > 0 else 0.0
        judge_f1 = (
            2 * judge_p * judge_r / (judge_p + judge_r)
            if (judge_p + judge_r) > 0
            else 0.0
        )

        # Agreement % (note-level approximation)
        total_agree = 0
        total_terms = 0
        for r in judge_rows:
            bt = int(r["baseline_tp"])
            bf = int(r["baseline_fp"])
            jt = int(r["judge_tp"])
            jf = int(r["judge_fp"])

            agree = min(bt, jt) + min(bf, jf)
            terms = bt + bf

            total_agree += agree
            total_terms += terms

        agreement_pct = (
            (total_agree / total_terms) * 100 if total_terms > 0 else 0.0
        )

        results.append(
            {
                "judge": judge,
                "baseline_p": round(baseline_p, 2),
                "baseline_r": round(baseline_r, 2),
                "baseline_f1": round(baseline_f1, 2),
                "judge_p": round(judge_p, 2),
                "judge_r": round(judge_r, 2),
                "judge_f1": round(judge_f1, 2),
                "agreement_pct": round(agreement_pct, 1),
            }
        )

    return results


def print_table(results):
    """Print results as formatted table to console."""
    header = (
        f"{'judge':<20} | {'Baseline_P':>10} | {'Baseline_R':>10} | {'Baseline_F1':>11} "
        f"| {'Judge_P':>8} | {'Judge_R':>8} | {'Judge_F1':>9} | {'Agreement%':>11}"
    )
    print("\n" + "=" * len(header))
    print("JUDGE VALIDATION AGGREGATED RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['judge']:<20} | {r['baseline_p']:>10} | {r['baseline_r']:>10} | {r['baseline_f1']:>11} "
            f"| {r['judge_p']:>8} | {r['judge_r']:>8} | {r['judge_f1']:>9} | {r['agreement_pct']:>10}%"
        )

    print("=" * len(header) + "\n")


def write_csv(results, output_path):
    """Write results to CSV file."""
    fieldnames = [
        "judge",
        "baseline_p",
        "baseline_r",
        "baseline_f1",
        "judge_p",
        "judge_r",
        "judge_f1",
        "agreement_pct",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Aggregated results written to {output_path}")


def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python aggregate_judge_validation.py <csv_folder> [output_file]")
    #     print("Example: python aggregate_judge_validation.py ./validation_results ./aggregated_results.csv")
    #     sys.exit(1)

    # csv_folder = sys.argv[1]
    # output_file = sys.argv[2] if len(sys.argv) > 2 else "aggregated_judge_validation.csv"
    csv_folder = r"C:\Users\wasim_xhy2aoh\Downloads\test_folder"
    output_file = r"C:\Users\wasim_xhy2aoh\Downloads\test_folder\aggregated_judge_validation.csv"

    if not os.path.isdir(csv_folder):
        print(f"Error: {csv_folder} is not a valid directory")
        sys.exit(1)

    rows = read_csvs_from_folder(csv_folder)
    results = compute_metrics(rows)
    print_table(results)
    write_csv(results, output_file)


if __name__ == "__main__":
    main()