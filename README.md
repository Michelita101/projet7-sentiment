# Projet 7 : Réalisez une analyse de sentiments grâce au Deep Learning

## Objectif du projet
Construire un pipeline NLP complet pour prédire automatiquement le **sentiment** (positif ou négatif) de tweets en anglais, incluant :
- Prétraitement avancé des textes
- Comparaison multi-embeddings (TF-IDF, Word2Vec, BERT…)
- Entraînement de modèles simples et avancés (MLP, BiLSTM, BERT)
- Déploiement d’une API de prédiction dans le cloud
- Suivi des performances via Azure & MLflow
- Création d’un blog pour expliciter la stratégie de monitoring

## Organisation du dépôt
```
├── BLOG.pages
├── conda_env_local.yml
├── data
│   ├── logo_MIC.png
│   ├── logo.png
│   ├── processed
│   │   ├── embeddings
│   │   ├── tweets_16k_clean_lem.parquet
│   │   ├── tweets_16k_clean_stem.parquet
│   │   ├── tweets_16k_raw.parquet
│   │   ├── tweets_16k.parquet
│   │   ├── tweets_balanced.parquet
│   │   ├── tweets_clean.parquet
│   │   └── tweets.parquet
│   └── raw
│       └── training.1600000.processed.noemoticon.csv
├── mlruns
│   ├── 0
│   │   └── meta.yaml
│   └── 687265510708116900
│       ├── 01d03e5f6f4d4fd785196a4a3776d296
│       ├── 1f67510d37624715a2bff84ff4f8630a
│       ├── 2d1b5c31cfb3465da7ec71781bfaadda
│       ├── 316f0255e8b14da98005289b1f93d50a
│       ├── 3462484987f3401a97dae030c7829b14
│       ├── 3a480c04bbf94802af6ddc6c1f05c75c
│       ├── 4a961a39ec2c4d5d9ac8ef8c49cabfc0
│       ├── 630c32d9a8524fe19e33db6076aa6ae6
│       ├── 6a50d712fdfb4ef0bb5cf160ed95883c
│       ├── bc68320029804df78621b3d275022d1a
│       ├── c796b169bd35487eae717c6153d3816f
│       ├── cdb88ebca75f4df7991d9f385beb5d51
│       ├── d4675218e0174901bd3279869f10a434
│       ├── d7134e3d4b51432cae6642cd65e70e88
│       ├── d88280759d794f519714c0862f756126
│       ├── e88fe0128a034e1da62538cd5622b88e
│       ├── ea6d27bcd60047c49e9e63f67d669640
│       ├── f8a92ed2ee13427d9eac00f33839f6d2
│       ├── meta.yaml
│       └── models
├── models
│   ├── bert
│   │   ├── bert_baseline_bs32_ep4.pt
│   │   └── bert_optim_bs16_ep3_lr1e-05.pt
│   ├── bert_sentiment
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   └── vocab.txt
│   ├── fasttext
│   │   └── wiki-news-300d-1M.vec
│   └── keras
│       ├── bilstm_keras_do0.5_bs32_ep10.h5
│       ├── bilstm_keras_opt_do0.5_bs64_ep15.h5
│       ├── mlp_count_do0.5_bs32_ep10.h5
│       └── mlp_tfidf_do0.5_bs32_ep10.h5
├── notebooks
│   ├── Dewerpe_Michèle_2_scripts_notebook_modélisation_092025.ipynb
│   └── Dewerpe_Michèle_2bis_scripts_notebook_modélisation_avancée_092025.ipynb
├── README.md
├── render.yaml
├── reports
│   ├── data_dictionary.csv
│   ├── data_dictionary.md
│   ├── figures
│   │   ├── bert_baseline_bs32_ep4
│   │   ├── bert_optim_bs16_ep3_lr1e-05
│   │   ├── bilstm_keras_do0.5_bs32_ep10
│   │   ├── bilstm_keras_opt_do0.5_bs64_ep15
│   │   ├── mlp_count_do0.5_bs32_ep10
│   │   └── mlp_tfidf_do0.5_bs32_ep10
│   └── logs
│       ├── api.log
│       ├── bert_baseline_bs32_ep4.txt
│       ├── bert_optim_bs16_ep3_lr1e-05.txt
│       ├── bilstm_keras_do0.5_bs32_ep10.txt
│       ├── bilstm_keras_opt_do0.5_bs64_ep15.txt
│       ├── mlp_count_do0.5_bs32_ep10.txt
│       ├── mlp_tfidf_do0.5_bs32_ep10.txt
│       └── results
├── requirements-deploy.txt
├── src
│   ├── __pycache__
│   │   ├── predict.cpython-311.pyc
│   │   └── serving_fastapi.cpython-311.pyc
│   ├── predict.py
│   ├── serving_fastapi.py
│   ├── tools
│   │   ├── __pycache__
│   │   ├── export_to_hf.py
│   │   └── README.md
│   ├── ui_streamlit.py
│   └── utils
│       ├── __pycache__
│       ├── tracking.py
│       └── transformers.py
└── tests
    ├── __pycache__
    │   ├── test_api.cpython-311-pytest-8.4.1.pyc
    │   └── test_predict.cpython-311-pytest-8.4.1.pyc
    ├── test_api.py
    └── test_predict.py
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
    python src/serving_fastapi.py
    ```
- Tester via une interface locale (ex. Streamlit) :
    ```bash
    streamlit run src/ui_streamlit.py
    ```

## Prétraitement NLP
- Nettoyage avancé : URLs, mentions, hashtags, ponctuation, répétitions, emojis
- Tokenisation, suppression des stopwords (hors négation)
- Double pipeline : stemming (Snowball) et lemmatisation (spaCy)
- Création d’un transformer `TextCleanerTransformer` compatible `Pipeline` / `GridSearchCV`
- Sauvegarde `.parquet` des versions `lem`, `stem`, `raw`
- Log complet dans MLflow avec artefacts

## Embeddings testés
- __TF-IDF__ et __CountVectorizer__ (`Scikit-learn`)
- __Word2Vec__ préentraîné (`Google News`, `300d`)
- __FastText__ préentraîné (`wiki-news-300d`)
- __BERT__ (`bert-base-uncased`, fine-tuné avec PyTorch + Transformers)
- Analyse de la __sparsité__, `np.save` des matrices vectorielles `.npy`

## Modélisation
- Entraînement de plusieurs modèles :
    - `LogisticRegression`, `SVC`, `NaiveBayes` (baselines)
    - `MLP` avec `Keras` (embeddings simples)
    - `BiLSTM` avec `Word2Vec` / `FastText`
    - `BERT` fine-tuné avec PyTorch
- Séparation `train/test` stratifiée
- Évaluation : `accuracy`, `precision`, `recall`, `f1`, `ROC-AUC`

## Industrialisation
- Classe `TextCleanerTransformer` sauvegardée dans `src/utils/transformers.py`
- Logs MLflow automatisés (`log_run`, `evaluate_model`, etc.)
- Export des modèles, logs, figures dans `models/`, `reports/`, `mlruns/`
- Visualisation des erreurs (`faux positifs`/ `faux négatifs`) avec emojis pour un rendu fun

## Mise en production
- API FastAPI déployée via __Render__
- Suivi de logs avec __Azure Application Insights__ (`opencensus`)
- Déploiement automatisé via __GitHub Actions__ (CI)
- Tests à venir (`pytest`)

## Résultats attendus
- Tableau comparatif des modèles (baseline vs `LSTM/BERT`, embeddings variés) avec accuracy, F1, AUC.
- Matrice de confusion et analyse des faux positifs/faux négatifs.
- Choix d’un modèle final après optimisation (hyperparamètres) et export du pipeline.
- Blog explicatif sur le suivi post-déploiement (monitoring, dérive, logs)

## Auteur
Projet réalisé par __Michèle Dewerpe__ dans le cadre du parcours _Ingénieur IA_ (OpenClassrooms).
