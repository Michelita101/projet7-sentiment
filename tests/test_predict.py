import requests

def test_predict_single_text():
    url = "http://127.0.0.1:8000/predict"
    payload = {"textes": "I absolutely love this product, it's amazing!"}
    response = requests.post(url, json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "prédictions" in data
    assert isinstance(data["prédictions"], list)
    assert data["prédictions"][0]["sentiment"] == "positif"  # attendu pour un texte positif


def test_predict_multiple_texts():
    url = "http://127.0.0.1:8000/predict"
    payload = {
        "textes": [
            "Fantastic experience, would recommend to everyone.",
            "Not bad at all",
            "Really good customer service!",
            "Terrible quality, very disappointed.",
            "Mediocre at best."
        ]
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "prédictions" in data
    assert isinstance(data["prédictions"], list)
    assert len(data["prédictions"]) == 5

    for pred in data["prédictions"]:
        assert pred["sentiment"] in ["positif", "négatif"]
