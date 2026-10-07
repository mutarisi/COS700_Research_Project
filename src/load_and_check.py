"""
Step 1: Load + sanity-check the CMU Keystroke Dynamics Benchmark Dataset.

Run directly: python3 src/load_and_check.py path/to/DSL-StrongPasswordData.csv
"""
import sys
import pandas as pd
import numpy as np

EXPECTED_SUBJECTS = 51
EXPECTED_SESSIONS = 8
EXPECTED_REPS_PER_SESSION = 50
EXPECTED_ROWS = EXPECTED_SUBJECTS * EXPECTED_SESSIONS * EXPECTED_REPS_PER_SESSION  # 20400


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def sanity_check(df: pd.DataFrame) -> dict:
    report = {}

    # 1. Shape
    report["n_rows"] = len(df)
    report["n_cols"] = df.shape[1]
    report["rows_match_expected"] = (len(df) == EXPECTED_ROWS)

    # 2. Subject / session structure
    id_cols = [c for c in ["subject", "sessionIndex", "rep"] if c in df.columns]
    report["id_columns_found"] = id_cols
    if "subject" in df.columns:
        n_subj = df["subject"].nunique()
        report["n_subjects"] = n_subj
        report["subjects_match_expected"] = (n_subj == EXPECTED_SUBJECTS)

        counts = df.groupby("subject").size()
        report["subjects_with_wrong_row_count"] = counts[counts != EXPECTED_SESSIONS * EXPECTED_REPS_PER_SESSION].to_dict()

    # 3. Missing values
    n_missing = df.isna().sum()
    report["columns_with_missing"] = n_missing[n_missing > 0].to_dict()

    # 4. Timing columns: identify H/DD/UD columns and check ranges
    timing_cols = [c for c in df.columns if c.startswith(("H.", "DD.", "UD."))]
    report["n_timing_columns"] = len(timing_cols)

    negative_counts = {}
    extreme_counts = {}
    for c in timing_cols:
        neg = (df[c] < 0).sum()
        if neg > 0:
            negative_counts[c] = int(neg)
        # sanity bound: keystroke hold/latency values should realistically be
        # under ~2 seconds; anything wildly beyond that suggests a data issue
        extreme = (df[c] > 2.0).sum()
        if extreme > 0:
            extreme_counts[c] = int(extreme)
    report["columns_with_negative_values"] = negative_counts
    report["columns_with_extreme_values_gt_2s"] = extreme_counts

    # 5. Duplicate rows
    report["n_duplicate_rows"] = int(df.duplicated().sum())

    return report


def print_report(report: dict):
    print("=" * 60)
    print("CMU DATASET SANITY CHECK")
    print("=" * 60)
    print(f"Rows: {report['n_rows']} (expected {EXPECTED_ROWS}) -> "
          f"{'OK' if report['rows_match_expected'] else 'MISMATCH'}")
    print(f"Columns: {report['n_cols']}")
    print(f"ID columns found: {report['id_columns_found']}")

    if "n_subjects" in report:
        print(f"Subjects: {report['n_subjects']} (expected {EXPECTED_SUBJECTS}) -> "
              f"{'OK' if report['subjects_match_expected'] else 'MISMATCH'}")
        if report["subjects_with_wrong_row_count"]:
            print(f"  WARNING: subjects with unexpected row counts: "
                  f"{report['subjects_with_wrong_row_count']}")
        else:
            print("  All subjects have the expected 400 reps.")

    print(f"Timing columns detected: {report['n_timing_columns']} (expected 31)")

    if report["columns_with_missing"]:
        print(f"WARNING: missing values found in: {report['columns_with_missing']}")
    else:
        print("No missing values.")

    if report["columns_with_negative_values"]:
        print(f"WARNING: negative timing values in: {report['columns_with_negative_values']}")
    else:
        print("No negative timing values.")

    if report["columns_with_extreme_values_gt_2s"]:
        print(f"WARNING: values > 2s (possible outliers/errors) in: "
              f"{report['columns_with_extreme_values_gt_2s']}")
    else:
        print("No extreme (>2s) timing values.")

    print(f"Duplicate rows: {report['n_duplicate_rows']}")
    print("=" * 60)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/DSL-StrongPasswordData.csv"
    df = load(path)
    report = sanity_check(df)
    print_report(report)