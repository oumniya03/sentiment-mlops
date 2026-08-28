#!/usr/bin/env python
"""
Soumet data/batch_test_sample.csv au batch endpoint et télécharge les résultats.
À placer à la racine du projet.
"""
import yaml
from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import AssetTypes
from azure.identity import AzureCliCredential



def load_config(path: str = "config.yml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    credential = AzureCliCredential(tenant_id="638a5759-8ea6-42b6-abfa-a0df4090efe1")
    ml_client = MLClient(
        credential,
        subscription_id=cfg["subscription_id"],
        resource_group_name=cfg["resource_group"],
        workspace_name=cfg["workspace_name"],
    )

    endpoint_name = cfg["batch_endpoint"]["name"]
    input_data = Input(path="data/batch_test_sample.csv", type=AssetTypes.URI_FILE)

    print("Soumission du job batch...")
    job = ml_client.batch_endpoints.invoke(
        endpoint_name=endpoint_name,
        input=input_data,
    )
    print(f"Job soumis : {job.name}")

    ml_client.jobs.stream(job.name)

    print("\nTéléchargement des résultats...")
    ml_client.jobs.download(name=job.name, download_path="./batch_results", output_name="score")
    print("Résultats téléchargés dans ./batch_results (cherche predictions.csv)")


if __name__ == "__main__":
    main()
