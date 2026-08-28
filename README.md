# Sentiment Classifier — MLOps sur Azure ML

Pipeline MLOps de bout en bout pour un classifieur de sentiment (positif/négatif) sur des tweets, avec tracking d'expériences, registre de modèle, déploiement via batch endpoint et monitoring de dérive.

Ce projet a été conçu volontairement **simple** (modèle léger, pas de deep learning) pour garantir un pipeline qui fonctionne réellement de bout en bout sur Azure ML, plutôt qu'un modèle plus sophistiqué mais dont le déploiement resterait bloqué.

## Architecture

```
Données (Sentiment140, sous-échantillon)
        │
        ▼
  Nettoyage texte (clean_text)
        │
        ▼
  TF-IDF (10k features) + Logistic Regression
  [entraînement local, tracking MLflow]
        │
        ▼
  Modèle enregistré dans le registre Azure ML
  (sentiment-classifier, sklearn.Pipeline complet)
        │
        ▼
  Batch Endpoint Azure ML
  (sentiment-batch-endpoint / sentiment-batch-deployment)
        │
        ▼
  Prédictions (predictions.csv)
        │
        ▼
  Monitoring de dérive (monitor.py — PSI + test KS)
```

## Résultats

Modèle : TF-IDF (10 000 features, unigrams + bigrams) → Régression logistique (`class_weight='balanced'`).

| Métrique | Valeur |
|---|---|
| Accuracy | 0.778 |
| Precision | 0.771 |
| Recall | 0.792 |
| F1 | 0.781 |
| AUC | 0.862 |

Entraînement et évaluation faits localement (quelques secondes, CPU) sur un sous-échantillon du dataset Sentiment140.

## Choix d'architecture : Batch Endpoint plutôt qu'Online Endpoint

La souscription Azure utilisée est un **compte gratuit personnel**, qui ne supporte pas les Managed Online Endpoints (erreur `SubscriptionNotRegistered` bloquante). Le déploiement a donc été fait via un **Batch Endpoint**, supporté sur toutes les souscriptions Azure. Ce choix est assumé comme une contrainte d'infrastructure documentée, pas comme un compromis technique sur la qualité du pipeline.

## Difficultés rencontrées et résolues

Le déploiement n'a pas fonctionné du premier coup — les problèmes suivants ont été identifiés et corrigés, dans l'ordre :

1. **Authentification (`Tenant mismatch: Token tenant does not match resource tenant`)** — bug connu avec les comptes Microsoft personnels (comptes de messagerie grand public) : l'appel programmatique `batch_endpoints.invoke()` (SDK Python et CLI) peut router l'authentification vers un tenant "consommateur" générique au lieu du tenant réel du workspace, même après `az login` explicite sur le bon tenant. **Contournement retenu** : invocation manuelle du job via le portail Azure ML Studio, qui utilise l'authentification du navigateur et n'est pas affecté par ce problème. Le script `test_batch_endpoint.py` reste présent dans le repo pour la documentation et une utilisation future en environnement CI/CD (compte de service), où ce problème ne se pose pas.

2. **`sklearn.exceptions.NotFittedError: idf vector is not fitted`** — le modèle chargeait sans erreur mais son état interne était corrompu. Cause : désalignement de version de scikit-learn entre l'environnement d'entraînement local et l'environnement Docker du batch endpoint (`conda.yml`). La désérialisation `joblib` d'un objet `TfidfVectorizer` entraîné avec une version peut donner un objet à l'état incohérent avec une version différente. **Fix** : figer les versions de `conda.yml` (scikit-learn, pandas, joblib, scipy) exactement sur celles de l'environnement d'entraînement local.

3. **Incompatibilités Python/librairies en cascade** — `scikit-learn==1.9.0` nécessite Python ≥ 3.11 ; `scipy==1.18.1` nécessite Python ≥ 3.12. `conda.yml` est passé de Python 3.10 → 3.11 → 3.12 pour satisfaire toutes les contraintes simultanément.

4. **`Could not find member 'deployment_name' on object of type 'BatchEndpointDefaults'`** — le SDK Python (`azure-ai-ml`) plante en essayant de définir le déploiement par défaut de l'endpoint via un objet `dict`, incohérence entre le modèle Python du SDK et le schéma REST actuel. **Contournement** : définir le déploiement par défaut via la CLI Azure (`az ml batch-endpoint update --set defaults.deployment_name=...`), qui gère correctement cette opération.

Ces problèmes ont été diagnostiqués via les logs `stderrlogs.txt` / `20_image_build_log.txt` du portail Azure ML Studio, en isolant à chaque itération le message d'erreur réel (souvent masqué par un `SystemExit: 42` générique en première lecture).

## Structure du projet

```
sentiment-mlops/
├── data/
│   ├── train.csv
│   ├── valid.csv
│   └── batch_test_sample.csv
├── src/
│   ├── train.py            # Entraînement local + tracking MLflow
│   ├── score.py             # Scoring local (test unitaire)
│   └── batch_score.py       # Scoring pour le batch endpoint (init/run)
├── pipelines/
│   ├── training_pipeline.py # Enregistrement du modèle dans le registre
│   └── deploy_batch.py      # Déploiement du batch endpoint
├── environments/
│   └── conda.yml            # Environnement du batch endpoint
├── models/
│   ├── model_pipeline.joblib   # sklearn.Pipeline complet (TF-IDF + LogReg)
│   ├── tfidf_vectorizer.joblib # Vectorizer seul (inspection/debug)
│   ├── logistic_model.joblib   # Modèle seul (inspection/debug)
│   └── metrics.json
├── generate_batch_sample.py
├── test_batch_endpoint.py
├── monitor.py                # Monitoring de dérive (PSI + test KS)
├── config.yml
└── .github/workflows/deploy.yml
```

## Reproduire le pipeline

```bash
# 1. Entraînement (local)
python src/train.py

# 2. Enregistrement du modèle dans Azure ML
python pipelines/training_pipeline.py --register-only

# 3. Génération d'un échantillon de test
python generate_batch_sample.py

# 4. Déploiement du batch endpoint
python pipelines/deploy_batch.py --model-version 1

# 5. Définir le déploiement par défaut (contournement CLI, voir ci-dessus)
az ml batch-endpoint update --name sentiment-batch-endpoint \
    --resource-group rg-sentiment-project --workspace-name sentiment-workspace \
    --set defaults.deployment_name=sentiment-batch-deployment

# 6. Tester (via le portail Azure ML Studio recommandé, cf. section authentification)
python test_batch_endpoint.py   # ou invocation manuelle via ml.azure.com

# 7. Monitoring de dérive sur de nouvelles données
python monitor.py --new-data <chemin_vers_nouvelles_donnees.csv>
```

## Monitoring de dérive

`monitor.py` compare la distribution de deux features simples dérivées du texte (longueur en caractères, nombre de mots) entre les données d'entraînement et de nouvelles données, via :
- **PSI (Population Stability Index)** — seuil d'alerte : PSI > 0.2
- **Test de Kolmogorov-Smirnov** — statistique et p-value

Validé sur un échantillon issu du même dataset (aucune dérive détectée, comme attendu) : PSI de 0.13 et 0.16 sur `char_len` et `word_count` respectivement.
