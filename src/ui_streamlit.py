import streamlit as st
import requests

# === Configuration ===
API_URL = "http://127.0.0.1:8000/predict"

# === Interface utilisateur ===
st.image("data/logo.png", width=200)
st.title("Analyse de sentiment")
st.write("Entrez un ou plusieurs tweets en anglais (un par ligne) 👇")

tweets_brut = st.text_area("Entrez un ou plusieurs tweets :", height=150)

if st.button("Analyser") and tweets_brut.strip():
    # Découpe en plusieurs lignes/tweets
    tweets = [t.strip() for t in tweets_brut.strip().split("\n") if t.strip()]
    données = {"textes": tweets}

    try:
        réponse = requests.post(API_URL, json=données)
        if réponse.status_code == 200:
            résultat = réponse.json()
            prédictions = résultat["prédictions"]

            # Création DataFrame pour affichage + export
            import pandas as pd
            df_résultats = pd.DataFrame(prédictions)

            # Affichage dynamique dans Streamlit
            st.subheader("Résultats de l’analyse")
            for _, row in df_résultats.iterrows():
                st.markdown(
                    f"__Tweet :__ {row['texte']}  \n"
                    f"Sentiment : __{row['sentiment']}__ ({row['confiance (%)']}%)"
                )

            # Section téléchargement
            st.subheader("Télécharger les résultats")
            csv = df_résultats.to_csv(index=False).encode("utf-8")
            json = df_résultats.to_json(orient="records", force_ascii=False, indent=2)

            st.download_button("⬇️ Télécharger en CSV", csv, "résultats_sentiments.csv", "text/csv")
            st.download_button("⬇️ Télécharger en JSON", json, "résultats_sentiments.json", "application/json")

        else:
            st.error(f"❌ Erreur {réponse.status_code} : {réponse.text}")
    except Exception as e:
        st.error(f"💥 Erreur lors de la requête : {e}")
