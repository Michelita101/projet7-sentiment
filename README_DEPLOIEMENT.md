# __Déploiement complet du projet 7 : Analyse de sentiments__

__Objectif__ : Activer toute la chaîne de déploiement (FastAPI, Streamlit, Azure, GitHub Actions, Render) étape par étape, accessible à un user non expert.  

Il s’adresse à toute personne souhaitant :
- ✅ vérifier que le déploiement fonctionne,
- 🔁 relancer manuellement un build,
- 🧪 tester les prédictions depuis l’API ou l’interface utilisateur,
- 🧠 comprendre le rôle de chaque service (GitHub, Render, Azure...).

---

## 🌐 1. Accéder aux services déployés

### 🔹 API FastAPI (via Render)
- URL publique :  
  👉 [https://bert-sentiment-api-m5zt.onrender.com](https://bert-sentiment-api-m5zt.onrender.com)

- Documentation Swagger :  
  👉 [https://bert-sentiment-api-m5zt.onrender.com/docs](https://bert-sentiment-api-m5zt.onrender.com/docs)

- Endpoint principal à tester : `/predict` (méthode POST)

---

### 🔹 Interface utilisateur Streamlit (via Render)
- URL publique :  
  👉 [https://projet7-sentiment-bnklr4gk5kwywdnpcgfzby.streamlit.app](https://projet7-sentiment-bnklr4gk5kwywdnpcgfzby.streamlit.app)

---

## 🛠️ 2. Vérifier que tout fonctionne

1. Ouvrir [https://bert-sentiment-api-m5zt.onrender.com/docs](https://bert-sentiment-api-m5zt.onrender.com/docs)
2. Cliquer sur __POST /predict__
3. Cliquer sur "Try it out"
4. Entrer un exemple de texte :
   ```json
   {
     "text": "This product is amazing! I love it 😍"
   }
   ````
5. Cliquez sur __Execute__ :
- Vous devez obtenir un score de probabilité (positive_proba) et une prédiction (label).

---

## 🔁 3. Relancer manuellement un build sur Render

Si un push GitHub n’a pas déclenché le déploiement, ou si le service ne répond plus :

1. Aller sur la page du service Render :
👉 [https://dashboard.render.com/web/srv-d2v8lf7diees73dshrkg](https://dashboard.render.com/web/srv-d2v8lf7diees73dshrkg)
2. Cliquer sur `Manual Deploy > Clear build cache & Deploy latest commit`
3. Attendre la fin du déploiement (logs visibles en temps réel)

---

## 🧪 4. Tester les prédictions

✅ __Depuis l’API FastAPI (Swagger UI)__
    - Voir section 2 ci-dessus pour `/predict`

✅ __Depuis l’interface Streamlit__
1. Ouvrir :
👉 [https://projet7-sentiment-bnklr4gk5kwywdnpcgfzby.streamlit.app](https://projet7-sentiment-bnklr4gk5kwywdnpcgfzby.streamlit.app)
2. Entrer un texte libre dans le champ de saisie
3. Cliquer sur __Analyser__
4. Lire le résultat de la prédiction (positif / négatif + confiance)

---

## 📈 5. Suivi des alertes et logs dans Azure

Le service est connecté à Azure Application Insights pour permettre :
- le suivi des logs du serveur FastAPI,
- l’identification des erreurs de production (ex. erreurs de prédiction, plantages),
- l’émission de warnings automatisés,
- l’analyse des performances (latence, nombre de requêtes, etc.).

✅ __Fonctionnement__  

L’API intègre un logger `AzureLogHandler` qui envoie les logs vers __Azure Application Insights__ si la variable `APPINSIGHTS_CONNECTION_STRING` est bien définie dans l’environnement.

🔔 __Alerte Azure en cas de prédictions douteuses__  

Une alerte est configurée dans __Azure Application Insights__ pour détecter des comportements anormaux du modèle :

- Si plus de `3` prédictions ont un score de confiance _inférieur à 60%_ (`confidence < 0.6`)  
- _dans une fenêtre glissante de 5 minutes_

__Objectif :__ prévenir des cas où le modèle hésite trop souvent, ce qui peut indiquer une dérive ou un problème de données.

__Recommandation :__ en cas d’alerte, vérifier les logs Application Insights et relancer une analyse métier ou technique.

👉 [Accéder à Application Insights sur Azure](https://portal.azure.com/#@micheledewerpeme.onmicrosoft.com/resource/subscriptions/e891789e-97c8-4e95-b530-ae6822e1d50f/resourceGroups/bert-sentiment-rg/providers/microsoft.insights/components/bert-sentiment-appinsights/overview)


---

## 🧠 6. Comprendre le rôle des services

| Service   | Rôle principal                                                     |
| --------- | ------------------------------------------------------------------ |
| 🔵 GitHub | Héberge le code source, déclenche les actions automatiques (CI/CD) |
| ☁️ Render | Héberge l’API (FastAPI) et l’interface (Streamlit)                 |
| 📈 Azure  | Collecte les logs de production (Application Insights)             |
| 🔬 MLflow | En local : suivi des expériences, hyperparamètres, scores          |
