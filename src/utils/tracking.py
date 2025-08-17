import mlflow
import mlflow.sklearn
from datetime import datetime

def log_run(model_name, params, metrics):
    """
    Fonction générique pour logger un run MLflow.
    Args:
        model_name (str): nom du modèle (ex: 'LogisticRegression')
        params (dict): dictionnaire d’hyperparamètres
        metrics (dict): dictionnaire de métriques (ex: {'accuracy': 0.85})
    """
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("timestamp", datetime.now().isoformat())
