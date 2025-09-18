from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import List, Union
from datetime import datetime
import uuid
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging
import os

# ================== ENVIRONNEMENT & LOGGING AZURE ==================

# Charger les variables d'environnement
load_dotenv()

# Récupérer la clé de connexion Application Insights
instrumentation_key = os.getenv("APPINSIGHTS_CONNECTION_STRING")

# Configurer le logger Azure
logger_azure = logging.getLogger("azure")
logger_azure.setLevel(logging.INFO)

if instrumentation_key:
    handler = AzureLogHandler(connection_string=instrumentation_key)
    logger_azure.addHandler(handler)
    logger_azure.info("Azure Application Insights logging is enabled.")
else:
    logger_azure.warning("APPINSIGHTS_CONNECTION_STRING not found. Azure logging is disabled.")

# ================== LOGGING LOCAL ==================

LOG_FILE = "reports/logs/api.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(message)s")

# ================== FASTAPI APP ==================

app = FastAPI(
    title="Sentiment Analysis API",
    description="Prédisez le sentiment d'un tweet.",
    version="1.0"
)

# ================== CHARGEMENT DU MODÈLE ==================

hf_token = os.getenv("HF_TOKEN")
MODEL_PATH = "models/bert_sentiment"

if hf_token:
    model_name = "Michelita/bert_sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# ================== SCHEMAS Pydantic ==================

class TweetInput(BaseModel):
    textes: Union[str, List[str]] = Field(..., description="Un tweet ou une liste de tweets")

class PredictionOutput(BaseModel):
    id: str
    horodatage: str
    prédictions: List[dict]

# ================== PRÉDICTION ==================

def prédire_sentiment(texte: str) -> dict:
    inputs = tokenizer(texte, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    proba = torch.softmax(logits, dim=1).detach().numpy()[0]
    label = int(proba.argmax())

    sentiment = "positif" if label == 1 else "négatif"
    confiance = round(float(proba[label]) * 100, 2)

    # === Si confiance faible, on loggue vers Azure ===
    if confiance < 60:
        logger_azure.warning(f"Prédiction douteuse : texte='{texte}' | sentiment='{sentiment}' | confiance={confiance}")

    return {
        "texte": texte,
        "sentiment": sentiment,
        "confiance (%)": confiance
    }

# ================== ROUTES ==================

@app.post("/predict", response_model=PredictionOutput)
async def predict(request: Request, données: TweetInput):
    id_requête = str(uuid.uuid4())
    horodatage = datetime.now().isoformat()

    # Normalisation
    textes = données.textes
    if isinstance(textes, str):
        textes = [textes]

    textes_valides = []
    erreurs = []

    for t in textes:
        if not isinstance(t, str):
            erreurs.append(f"Entrée invalide : {t}")
        elif len(t.strip()) == 0:
            erreurs.append("Le texte ne peut pas être vide.")
        elif len(t.strip()) > 280:
            erreurs.append("Le texte dépasse 280 caractères.")
        else:
            textes_valides.append(t.strip())

    if not textes_valides:
        return PredictionOutput(
            id=id_requête,
            horodatage=horodatage,
            prédictions=[{"texte": None, "sentiment": "erreur", "confiance (%)": 0}]
        )

    prédictions = [prédire_sentiment(t) for t in textes_valides]

    logging.info(f"{id_requête} - {len(prédictions)} prédictions - OK")

    return PredictionOutput(
        id=id_requête,
        horodatage=horodatage,
        prédictions=prédictions
    )

@app.get("/")
def read_root():
    return {"message": "API opérationnelle"}
