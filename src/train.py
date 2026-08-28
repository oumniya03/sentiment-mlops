#!/usr/bin/env python
"""
Entraînement local : TF-IDF (max 10k features) -> Régression logistique.
Sauvegarde vectorizer + modèle ensemble (joblib).
Tracking MLflow : accuracy, precision, recall, F1, AUC, hyperparamètres.
"""
import argparse
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline


def clean_text(text: str) -> str:
    """Nettoyage basique : lowercase, suppression URL/mentions/ponctuation."""
    import re
    text = text.lower()
    # URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Mentions @user
    text = re.sub(r'@\w+', '', text)
    # Ponctuation (garde lettres, chiffres, espaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_data(train_path: Path, valid_path: Path) -> tuple:
    """Charge train.csv et valid.csv."""
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    X_train = train_df['text'].astype(str).apply(clean_text)
    y_train = train_df['label'].values

    X_valid = valid_df['text'].astype(str).apply(clean_text)
    y_valid = valid_df['label'].values

    return X_train, y_train, X_valid, y_valid


def train_model(
    X_train, y_train,
    X_valid, y_valid,
    max_features: int = 10000,
    C: float = 1.0,
    random_state: int = 42,
    max_iter: int = 1000
) -> tuple:
    """
    Entraîne TF-IDF + LogisticRegression.
    Retourne (pipeline, vectorizer, model, metrics_dict).
    """
    print(f"Entraînement : max_features={max_features}, C={C}")

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # unigrams + bigrams
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    # Logistic Regression
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
        solver='lbfgs',
        class_weight='balanced'
    )

    # Pipeline
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('clf', model)
    ])

    # Entraînement
    pipeline.fit(X_train, y_train)

    # Prédictions
    y_pred = pipeline.predict(X_valid)
    y_proba = pipeline.predict_proba(X_valid)[:, 1]

    # Métriques
    metrics = {
        'accuracy': float(accuracy_score(y_valid, y_pred)),
        'precision': float(precision_score(y_valid, y_pred)),
        'recall': float(recall_score(y_valid, y_pred)),
        'f1': float(f1_score(y_valid, y_pred)),
        'auc': float(roc_auc_score(y_valid, y_proba))
    }

    # Vérifier taille vecteur fixe
    sample_vec = vectorizer.transform([X_train.iloc[0]])
    vector_dim = sample_vec.shape[1]
    print(f"  Dimension vecteur TF-IDF : {vector_dim} (attendu: {max_features})")
    assert vector_dim <= max_features, f"Dimension inattendue: {vector_dim}"

    print(f"  Metrics: {json.dumps(metrics, indent=2)}")

    return pipeline, vectorizer, model, metrics


def save_model(pipeline, vectorizer, model, output_dir: Path):
    """Sauvegarde le pipeline complet (vectorizer + modèle) et aussi séparément."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline complet (recommandé pour le scoring)
    pipeline_path = output_dir / "model_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    print(f"OK Pipeline sauvegardé : {pipeline_path}")

    # Vectorizer seul (pour inspection/debug)
    vectorizer_path = output_dir / "tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    print(f"OK Vectorizer sauvegardé : {vectorizer_path}")

    # Modèle seul
    model_path = output_dir / "logistic_model.joblib"
    joblib.dump(model, model_path)
    print(f"OK Modèle sauvegardé : {model_path}")

    return pipeline_path, vectorizer_path, model_path


def main():
    parser = argparse.ArgumentParser(description="Entraîne TF-IDF + LogisticRegression sur Sentiment140")
    parser.add_argument("--train-data", type=str, default="data/train.csv", help="Chemin train.csv")
    parser.add_argument("--valid-data", type=str, default="data/valid.csv", help="Chemin valid.csv")
    parser.add_argument("--output-dir", type=str, default="models", help="Dossier de sortie")
    parser.add_argument("--max-features", type=int, default=10000, help="Max features TF-IDF")
    parser.add_argument("--C", type=float, default=1.0, help="Régularisation inverse LogisticRegression")
    parser.add_argument("--max-iter", type=int, default=1000, help="Max iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mlflow-uri", type=str, default="mlruns", help="MLflow tracking URI")
    parser.add_argument("--experiment", type=str, default="sentiment-training", help="Nom expérience MLflow")
    parser.add_argument("--run-name", type=str, default=None, help="Nom du run MLflow")
    args = parser.parse_args()

    # MLflow setup
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    # Charger données
    X_train, y_train, X_valid, y_valid = load_data(
        Path(args.train_data), Path(args.valid_data)
    )

    # Run MLflow
    with mlflow.start_run(run_name=args.run_name):
        # Log hyperparams
        mlflow.log_params({
            'max_features': args.max_features,
            'C': args.C,
            'max_iter': args.max_iter,
            'random_state': args.seed,
            'ngram_range': '(1,2)',
            'min_df': 2,
            'max_df': 0.95,
            'sublinear_tf': True,
            'solver': 'lbfgs',
            'class_weight': 'balanced'
        })

        # Entraîner
        pipeline, vectorizer, model, metrics = train_model(
            X_train, y_train, X_valid, y_valid,
            max_features=args.max_features,
            C=args.C,
            random_state=args.seed,
            max_iter=args.max_iter
        )

        # Log metrics
        mlflow.log_metrics(metrics)

        # Sauvegarder localement
        output_dir = Path(args.output_dir)
        save_model(pipeline, vectorizer, model, output_dir)

        # Log model dans MLflow (pipeline complet)
        mlflow.sklearn.log_model(pipeline, "model")

        # Sauvegarder métriques en JSON
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"OK Métriques sauvées : {metrics_path}")

        print(f"\n✅ Entraînement terminé !")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")
        print(f"   Métriques: {metrics}")


if __name__ == "__main__":
    main()