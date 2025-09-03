# src/predict.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Chemin vers ton modèle Hugging Face sauvegardé
MODEL_PATH = "models/bert_sentiment"

# Chargement du tokenizer et du modèle
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # mode évaluation

def predict_sentiment(text: str) -> dict:
    """
    Prédit le sentiment d'un tweet.
    Retourne un dictionnaire avec le label (0 ou 1), le pourcentage, et le texte.
    """
    # Encodage du texte
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Prédiction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probas = torch.nn.functional.softmax(logits, dim=1)
        pred = torch.argmax(probas, axis=1).item()
        confidence = round(probas[0][pred].item() * 100, 2)

    # Interprétation du label
    label_str = "positif" if pred == 1 else "négatif"

    return {
        "texte": text,
        "sentiment": label_str,
        "confiance (%)": confidence
    }
