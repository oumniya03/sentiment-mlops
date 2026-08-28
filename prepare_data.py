#!/usr/bin/env python
"""
Prépare les données Sentiment140 pour l'entraînement.
Télécharge depuis Stanford, prend un sous-échantillon équilibré (20-50k lignes),
et crée train.csv / valid.csv.
"""
import argparse
import pandas as pd
from pathlib import Path
import urllib.request
import zipfile
import io
import ssl

# Désactiver la vérification SSL pour Stanford
ssl._create_default_https_context = ssl._create_unverified_context

STANFORD_URL = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"


def download_and_extract(output_dir: Path) -> Path:
    """Télécharge et extrait le ZIP Stanford."""
    csv_path = output_dir / "training.1600000.processed.noemoticon.csv"
    if csv_path.exists():
        print(f"OK Fichier deja present : {csv_path}")
        return csv_path

    print("Telechargement de Sentiment140 depuis Stanford (~ 80 MB)...")
    req = urllib.request.Request(STANFORD_URL, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req, timeout=120)
    zip_data = response.read()
    print(f"  ZIP telecharge : {len(zip_data) / 1e6:.1f} MB")

    print("Extraction...")
    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
        # Le ZIP contient training.1600000.processed.noemoticon.csv
        for name in z.namelist():
            if name.endswith('.csv') and 'training' in name:
                z.extract(name, output_dir)
                extracted = output_dir / name
                if extracted != csv_path:
                    extracted.rename(csv_path)
                print(f"OK Extrait : {csv_path}")
                return csv_path

    raise FileNotFoundError("Fichier CSV non trouve dans le ZIP")


def prepare_data(
    input_csv: Path,
    output_dir: Path,
    n_samples: int = 40000,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[Path, Path]:
    """
    Crée un sous-échantillon équilibré et le split train/valid.
    Retourne (train_path, valid_path).
    """
    # Colonnes Sentiment140 : target, id, date, flag, user, text
    COLUMNS = ['target', 'id', 'date', 'flag', 'user', 'text']

    print(f"Chargement de {input_csv}...")
    df = pd.read_csv(input_csv, encoding='latin-1', names=COLUMNS, header=None)
    print(f"  Total lignes : {len(df):,}")

    # Mapper target: 0 -> 0 (negatif), 4 -> 1 (positif)
    df['label'] = (df['target'] == 4).astype(int)

    # Garder seulement text et label
    df = df[['text', 'label']].copy()

    # Nettoyage basique : retirer lignes vides
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].str.len() > 0].copy()

    # Équilibrer les classes
    n_per_class = n_samples // 2
    neg = df[df['label'] == 0].sample(n=min(n_per_class, len(df[df['label'] == 0])), random_state=random_state)
    pos = df[df['label'] == 1].sample(n=min(n_per_class, len(df[df['label'] == 1])), random_state=random_state)

    balanced = pd.concat([neg, pos]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"  Sous-echantillon equilibre : {len(balanced):,} lignes ({len(neg)} neg + {len(pos)} pos)")

    # Split train/valid
    from sklearn.model_selection import train_test_split
    train_df, valid_df = train_test_split(
        balanced, test_size=test_size, random_state=random_state, stratify=balanced['label']
    )

    train_path = output_dir / "train.csv"
    valid_path = output_dir / "valid.csv"

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)

    print(f"OK Train : {train_path} ({len(train_df):,} lignes)")
    print(f"OK Valid : {valid_path} ({len(valid_df):,} lignes)")

    # Stats rapides
    print(f"\nDistribution train :")
    print(train_df['label'].value_counts().sort_index())
    print(f"\nDistribution valid :")
    print(valid_df['label'].value_counts().sort_index())

    return train_path, valid_path


def main():
    parser = argparse.ArgumentParser(description="Prépare les données Sentiment140")
    parser.add_argument("--n-samples", type=int, default=40000, help="Taille du sous-echantillon total (defaut: 40000)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion validation (defaut: 0.2)")
    parser.add_argument("--data-dir", type=str, default="data", help="Dossier de sortie (defaut: data/)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (defaut: 42)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = download_and_extract(data_dir)
    prepare_data(csv_path, data_dir, n_samples=args.n_samples, test_size=args.test_size, random_state=args.seed)

    print("\nOK Donnees pretes !")


if __name__ == "__main__":
    main()