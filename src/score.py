#!/usr/bin/env python
"""
Scoring local (style online) : charge le pipeline et prédit sur des textes.
Usage: python src/score.py --model models/model_pipeline.joblib --text "I love this!" "This is terrible"
"""
import argparse
import joblib
import numpy as np
from pathlib import Path


def predict(texts, model_path: Path):
    """Prédit le sentiment pour une liste de textes."""
    print(f"Chargement du modèle depuis {model_path}...")
    pipeline = joblib.load(model_path)

    # Prédiction
    preds = pipeline.predict(texts)
    probas = pipeline.predict_proba(texts)[:, 1]

    results = []
    for text, pred, proba in zip(texts, preds, probas):
        label = "positif" if pred == 1 else "négatif"
        results.append({
            'text': text,
            'prediction': int(pred),
            'label': label,
            'probability': float(proba)
        })
        print(f"  [{label:8s} | p={proba:.4f}] {text[:80]}...")

    return results


def main():
    parser = argparse.ArgumentParser(description="Scoring local avec le modèle entraîné")
    parser.add_argument("--model", type=str, default="models/model_pipeline.joblib", help="Chemin vers le pipeline joblib")
    parser.add_argument("--text", nargs='+', help="Textes à classifier (si absent, mode interactif)")
    parser.add_argument("--file", type=str, help="Fichier CSV avec colonne 'text' à scorer")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERREUR: Modèle non trouvé : {model_path}")
        print("Entraînez d'abord avec: python src/train.py")
        return 1

    if args.file:
        import pandas as pd
        df = pd.read_csv(args.file)
        texts = df['text'].astype(str).tolist()
        print(f"Scoring de {len(texts)} textes depuis {args.file}...")
        predict(texts, model_path)
    elif args.text:
        predict(args.text, model_path)
    else:
        # Mode interactif
        print("Mode interactif (Ctrl+C pour quitter)")
        print(f"Modèle chargé: {model_path}")
        while True:
            try:
                text = input("\nEntrez un texte: ").strip()
                if text:
                    predict([text], model_path)
            except KeyboardInterrupt:
                print("\nAu revoir !")
                break
            except EOFError:
                break

    return 0


if __name__ == "__main__":
    exit(main())