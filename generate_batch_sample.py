#!/usr/bin/env python
"""
Génère un petit fichier d'exemple à partir de data/valid.csv pour tester le batch endpoint.
À placer à la racine du projet.
"""
import pandas as pd
from pathlib import Path


def main():
    valid_df = pd.read_csv("data/valid.csv")

    n = min(50, len(valid_df))
    sample = valid_df.sample(n=n, random_state=42)[["text"]]

    out_path = Path("data/batch_test_sample.csv")
    sample.to_csv(out_path, index=False)
    print(f"Sauvegardé : {out_path} ({len(sample)} lignes)")


if __name__ == "__main__":
    main()
