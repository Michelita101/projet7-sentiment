# 📁 tools/

Ce dossier contient des **scripts ponctuels** et **utilitaires**, utilisés pour des actions spécifiques en dehors du pipeline principal de modélisation ou d’inférence.

## Fichiers présents

- `export_to_hf.py`  
  Script d'export vers le **Hugging Face Hub**.  
  Utilisé en fin de projet pour uploader un modèle entraîné, ses métadonnées, et éventuellement des artefacts.

## Bonnes pratiques

- Ne pas importer ces scripts dans le code de production (ex: `src/`), sauf si nécessaire.
- Ajouter un script ici uniquement s'il répond à un besoin **ponctuel**, **terminal** ou **hors-pipeline** (packaging, upload, génération de documentation…).
- Préfixer le nom par le verbe d’action (`export_`, `generate_`, `build_`, etc.) si possible.

