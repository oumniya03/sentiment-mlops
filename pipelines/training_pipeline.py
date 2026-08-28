#!/usr/bin/env python
"""
Enregistre le modèle déjà entraîné localement (models/) dans le registre Azure ML.
À placer dans pipelines/training_pipeline.py

On n'entraîne PAS sur Azure ML compute ici : le modèle est déjà entraîné en local
(TF-IDF + LogReg, léger, quelques secondes) — pas besoin de consommer du crédit
compute pour ça. Ce script se contente d'enregistrer les artefacts.
"""
import argparse
import yaml
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import AzureCliCredential


def load_config(path: str = "config.yml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_ml_client(cfg: dict) -> MLClient:
    credential = AzureCliCredential(tenant_id="638a5759-8ea6-42b6-abfa-a0df4090efe1")
    return MLClient(
        credential,
        subscription_id=cfg["subscription_id"],
        resource_group_name=cfg["resource_group"],
        workspace_name=cfg["workspace_name"],
    )


def register_model(ml_client: MLClient, cfg: dict):
    model = Model(
        path=cfg["model"]["local_path"],       # dossier "models" entier (pipeline + vectorizer + metrics)
        type=AssetTypes.CUSTOM_MODEL,
        name=cfg["model"]["name"],
        description="Sentiment classifier : TF-IDF + Logistic Regression (Sentiment140, sous-échantillon).",
    )
    registered = ml_client.models.create_or_update(model)
    print(f"Modèle enregistré : {registered.name} v{registered.version}")
    return registered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-only", action="store_true", help="N'enregistre que le modèle (comportement actuel par défaut)")
    args = parser.parse_args()

    cfg = load_config()
    ml_client = get_ml_client(cfg)
    register_model(ml_client, cfg)


if __name__ == "__main__":
    main()
