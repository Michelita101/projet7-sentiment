import requests

def test_api_responds():
    # Remplace par l'URL de ton API déployée plus tard
    url = "http://127.0.0.1:8000/"
    response = requests.get(url)
    assert response.status_code == 200
    assert "message" in response.json()