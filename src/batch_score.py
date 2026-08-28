#!/usr/bin/env python
"""
Script de scoring pour Azure ML Batch Endpoint.
À placer dans src/batch_score.py

IMPORTANT : reproduit exactement clean_text() de train.py, car le pipeline
a été entraîné sur du texte déjà nettoyé (clean_text n'est PAS inclus dans
le sklearn.Pipeline lui-même).
"""
import os
import re
import glob
import joblib
import pandas as pd


def clean_text(text: str) -> str:
    """Identique à train.py — NE PAS modifier sans réentraîner le modèle."""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def init():
    global pipeline
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    if model_dir is None:
        raise EnvironmentError("AZUREML_MODEL_DIR non défini — ce script doit tourner dans Azure ML.")

    # Le modèle enregistré peut être niché dans un sous-dossier (ex: models/model_pipeline.joblib)
    matches = glob.glob(os.path.join(model_dir, "**", "model_pipeline.joblib"), recursive=True)
    if not matches:
        raise FileNotFoundError(f"model_pipeline.joblib introuvable sous {model_dir}")

    pipeline = joblib.load(matches[0])
    print(f"Pipeline sentiment chargé depuis {matches[0]}")


def run(mini_batch):
    """
    mini_batch : liste de chemins de fichiers (CSV attendu, colonne 'text').
    Retourne un DataFrame avec text, prediction, probability_positive.
    """
    results = []

    for file_path in mini_batch:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            print(f"Format non supporté, ignoré : {file_path}")
            continue

        if "text" not in df.columns:
            raise ValueError(f"Colonne 'text' manquante dans {file_path}")

        raw_texts = df["text"].astype(str).tolist()
        cleaned_texts = [clean_text(t) for t in raw_texts]

        predictions = pipeline.predict(cleaned_texts)
        probabilities = pipeline.predict_proba(cleaned_texts)[:, 1]

        out = pd.DataFrame({
            "text": raw_texts,
            "prediction": predictions,
            "probability_positive": probabilities
        })
        results.append(out)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame(columns=["text", "prediction", "probability_positive"])
