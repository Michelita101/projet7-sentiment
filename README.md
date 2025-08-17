# Projet 7 : Réalisez une analyse de sentiments grâce au Deep Learning

## Objectif du projet
Ce projet consiste à développer un modèle de deep learning capable de prédire le sentiment (positif ou négatif) de tweets en anglais.  
Le livrable inclut :
- un pipeline complet de traitement de texte et de modélisation,
- une comparaison de plusieurs embeddings et architectures,
- une API déployée dans le cloud pour exposer le modèle,
- un suivi de performance en production via Azure Application Insight,
- un blog expliquant la stratégie de suivi.

## Organisation du dépôt
```
├── data/ # Données (raw non versionnées, processed versionnées si légères)
│ ├── raw/ # Données brutes (gros fichiers, ignorés par Git)
│ └── processed/ # Données nettoyées/extraites (tweets.parquet, etc.)
├── notebooks/ # Notebooks d'exploration et de modélisation
├── reports/ # Slides de soutenance, figures, graphiques
├── src/ # Scripts Python réutilisables
├── mlruns/ # Logs MLflow (expérimentations)
├── README.md # Ce document
└── environment.yml # Fichier pour recréer l'environnement conda
```

## Installation
1. Clôner le dépôt :
   ```bash
   git clone https://github.com/username/projet7-sentiment.git
   cd projet7-sentiment
   ```
2. Créer l'environnement conda :
```bash
conda env create -f environment.yml
conda activate sentiment-oc
```

## Utilisation
- Ouvrir les notebooks d’exploration et de modélisation :
    ```bash
    jupyter lab
    ```
- Suivre et comparer les expériences avec MLflow (UI locale) :
    ```bash
    mlflow ui --backend-store-uri ./mlruns --port 5000
    ```
- Lancer l’API de prédiction (exemple générique ; adapter au script réel du projet) :
    ```bash
    python src/api/app.py
    ```
- Tester via une interface locale (ex. Streamlit) :
    ```bash
    streamlit run src/app/streamlit_app.py
    ```

## Suivi des expériences
- Tracking avec __MLflow__ : scores, autres métriques (ex. `AUC`, `F1`), hyperparamètres, temps d’entraînement/prédiction, artefacts (`courbes ROC`/`PR`, `matrices de confusion`), modèles sérialisés.
- Standardisation du tracking via une fonction dédiée appelée à chaque entraînement.

## Mise en production
- __API__ packagée (ex. `Flask/FastAPI`) et déployée via CI/CD (GitHub Actions) sur une cible cloud (ex. Azure WebApp plan gratuit F1, ou équivalent).
- Tests unitaires exécutés automatiquement dans le pipeline de déploiement.
- Modèle exporté au format TensorFlow (model.save(...)) ou équivalent adapté à l’inférence.

## Résultats attendus
- Tableau comparatif des modèles (baseline vs `LSTM/BERT`, embeddings variés) avec accuracy, F1, AUC.
- Matrice de confusion et analyse des faux positifs/faux négatifs.
- Choix d’un modèle final après optimisation (hyperparamètres) et export du pipeline.

## Auteur
Projet réalisé par __Michèle Dewerpe__ dans le cadre du parcours _Ingénieur IA_ (OpenClassrooms).