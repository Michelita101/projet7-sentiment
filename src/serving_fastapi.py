from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import List, Union
from datetime import datetime
import uuid
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# ========== CONFIGURATION ==========
MODEL_PATH = "models/bert_sentiment"
LOG_FILE = "reports/logs/api.log"

# ========== INITIALISATION ==========
app = FastAPI(title="Sentiment Analysis API", description="Prédisez le sentiment d'un tweet.", version="1.0")

# Chargement modèle/tokenizer local ou distant via Hugging Face
hf_token = os.getenv("HF_TOKEN")

# Utilise le modèle local si pas de token, sinon Hugging Face
if hf_token:
    model_name = "Michelita101/bert_sentiment"  # change ce nom si besoin
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# ========== LOGGING ==========
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(message)s")

# ========== CLASSES DE DONNÉES ==========
class TweetInput(BaseModel):
    textes: Union[str, List[str]] = Field(..., description="Un tweet ou une liste de tweets")

class PredictionOutput(BaseModel):
    id: str
    horodatage: str
    prédictions: List[dict]

# ========== UTILITAIRES ==========
def prédire_sentiment(texte: str) -> dict:
    inputs = tokenizer(texte, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    proba = torch.softmax(logits, dim=1).detach().numpy()[0]
    label = int(proba.argmax())

    sentiment = "positif" if label == 1 else "négatif"
    confiance = round(float(proba[label]) * 100, 2)

    return {
        "texte": texte,
        "sentiment": sentiment,
        "confiance (%)": confiance
    }

# ========== ROUTE PRINCIPALE ==========
@app.post("/predict", response_model=PredictionOutput)
async def predict(request: Request, données: TweetInput):
    id_requête = str(uuid.uuid4())
    horodatage = datetime.now().isoformat()

    # Normalisation des données
    textes = données.textes
    if isinstance(textes, str):
        textes = [textes]

    # Vérifications
    erreurs = []
    textes_valides = []
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
    
    # Prédictions
    prédictions = [prédire_sentiment(t) for t in textes_valides]

    # Logging
    logging.info(f"{id_requête} - {len(prédictions)} prédictions - OK")

    return PredictionOutput(
        id=id_requête,
        horodatage=horodatage,
        prédictions=prédictions
    )
# ========== ROUTE DE TEST ==========
@app.get("/")
def read_root():
    return {"message": "API opérationnelle"}   
    