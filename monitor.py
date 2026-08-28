#!/usr/bin/env python
"""
Monitoring de dérive : compare la distribution des nouvelles données à celle du train,
sur des features simples dérivées du texte (longueur, nombre de mots).
À placer à la racine du projet.

Usage :
    python monitor.py --new-data data/batch_test_sample.csv
"""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index entre deux distributions."""
    breakpoints = np.linspace(0, 100, buckets + 1)
    bucket_edges = np.percentile(expected, breakpoints)
    bucket_edges[0] -= 1e-6
    bucket_edges[-1] += 1e-6

    expected_counts = np.histogram(expected, bins=bucket_edges)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=bucket_edges)[0] / len(actual)

    expected_counts = np.where(expected_counts == 0, 1e-6, expected_counts)
    actual_counts = np.where(actual_counts == 0, 1e-6, actual_counts)

    psi = np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts))
    return float(psi)


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["char_len"] = df["text"].astype(str).str.len()
    df["word_count"] = df["text"].astype(str).str.split().apply(len)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, default="data/train.csv")
    parser.add_argument("--new-data", type=str, required=True)
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    parser.add_argument("--output", type=str, default="drift_report.json")
    args = parser.parse_args()

    train_df = add_text_features(pd.read_csv(args.train_data))
    new_df = add_text_features(pd.read_csv(args.new_data))

    report = {}
    for feature in ["char_len", "word_count"]:
        psi = compute_psi(train_df[feature].values, new_df[feature].values)
        ks_stat, ks_pval = ks_2samp(train_df[feature].values, new_df[feature].values)
        report[feature] = {
            "psi": round(psi, 4),
            "drift_detected": bool(psi > args.psi_threshold),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pval), 4),
        }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    any_drift = any(v["drift_detected"] for v in report.values())
    print(f"\n{'⚠️  DÉRIVE DÉTECTÉE' if any_drift else '✅ Pas de dérive significative'} (seuil PSI={args.psi_threshold})")


if __name__ == "__main__":
    main()
