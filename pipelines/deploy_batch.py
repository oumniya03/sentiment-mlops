#!/usr/bin/env python
"""
Crée/déploie le Batch Endpoint + Batch Deployment pour le modèle sentiment.
À placer dans pipelines/deploy_batch.py
"""
import argparse
import yaml
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    BatchEndpoint,
    ModelBatchDeployment,
    ModelBatchDeploymentSettings,
    Environment,
)
from azure.ai.ml.constants import BatchDeploymentOutputAction
from azure.identity import DefaultAzureCredential


def load_config(path: str = "config.yml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_ml_client(cfg: dict) -> MLClient:
    credential = DefaultAzureCredential()
    return MLClient(
        credential,
        subscription_id=cfg["subscription_id"],
        resource_group_name=cfg["resource_group"],
        workspace_name=cfg["workspace_name"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", type=str, default="1")
    args = parser.parse_args()

    cfg = load_config()
    ml_client = get_ml_client(cfg)

    endpoint_name = cfg["batch_endpoint"]["name"]
    deployment_name = cfg["batch_endpoint"]["deployment_name"]
    model_name = cfg["model"]["name"]
    compute_name = cfg["compute_cluster"]

    # --- Endpoint (idempotent) ---
    try:
        endpoint = ml_client.batch_endpoints.get(endpoint_name)
        print(f"Endpoint existant récupéré : {endpoint_name}")
    except Exception:
        endpoint = BatchEndpoint(name=endpoint_name, description="Batch endpoint sentiment classifier")
        ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
        print(f"Endpoint créé : {endpoint_name}")

    # --- Modèle enregistré ---
    model = ml_client.models.get(name=model_name, version=args.model_version)
    print(f"Modèle récupéré : {model.name} v{model.version}")

    # --- Environnement (créé inline depuis conda.yml, pas besoin de pré-enregistrer) ---
    environment = Environment(
        name="sentiment-env",
        conda_file="environments/conda.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    )

    # --- Déploiement ---
    deployment = ModelBatchDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        code_path="src",
        scoring_script="batch_score.py",
        environment=environment,
        compute=compute_name,
        settings=ModelBatchDeploymentSettings(
            instance_count=1,
            max_concurrency_per_instance=1,
            mini_batch_size=10,
            output_action=BatchDeploymentOutputAction.APPEND_ROW,
            output_file_name="predictions.csv",
        ),
    )
    ml_client.batch_deployments.begin_create_or_update(deployment).result()
    print(f"Déploiement créé : {deployment_name}")

    # --- Définir le déploiement par défaut ---
    # NB : le SDK Python (azure-ai-ml) plante systématiquement sur cette étape
    # ("Could not find member 'deployment_name' on object of type
    # 'BatchEndpointDefaults'") à cause d'une incohérence entre le modèle Python
    # et le schéma REST actuel. On utilise donc la CLI Azure, qui gère cette
    # opération correctement :
    #
    #   az ml batch-endpoint update --name <endpoint_name> \
    #       --resource-group <resource_group> --workspace-name <workspace_name> \
    #       --set defaults.deployment_name=<deployment_name>
    #
    # À lancer manuellement une fois après ce script (ou à automatiser via
    # subprocess si besoin en CI/CD).
    print(f"\nDéploiement créé : {deployment_name}")
    print("Pour le définir comme déploiement par défaut, lance :")
    print(
        f"  az ml batch-endpoint update --name {endpoint_name} "
        f"--resource-group {cfg['resource_group']} --workspace-name {cfg['workspace_name']} "
        f"--set defaults.deployment_name={deployment_name}"
    )
    print(f"\nBatch endpoint prêt : {endpoint_name}")


if __name__ == "__main__":
    main()