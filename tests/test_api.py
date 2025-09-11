# test mis à jour pour relancer CI

import requests

def test_api_responds():
    # Remplacer par l'URL de l'API déployée plus tard si nécessaire
    url = "http://127.0.0.1:8000/"
    response = requests.get(url)
    assert response.status_code == 200
    assert "message" in response.json()