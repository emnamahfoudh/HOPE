# Analyse de Sentiment Multilingue Tunisien — TOPNET

Projet de Fin d'Études (PFE) 2025–2026

Système complet d'analyse de sentiment pour les commentaires clients tunisiens (arabe, arabizi, français, anglais), basé sur **MARBERT** fine-tuné, avec explicabilité **LIME** et tableau de bord **Streamlit**.

---

## Architecture du pipeline

```
Données brutes (CSV / JSON)
        ↓
Nettoyage avancé (nettoyage_avance.ipynb)
        ↓
Détection de langue (arabe / arabizi / fr / en)
        ↓
Traduction → arabe (Groq API — llama-3.3-70b)
        ↓
Classification MARBERT (3 classes : positif / neutre / négatif)
        ↓
Dashboard Streamlit + Explicabilité LIME
```

---

## Structure du projet

```
HOPE/
├── app.py                      # Application Streamlit (dashboard)
├── run_predictions.py          # Pipeline de prédiction batch
├── nettoyage_avance.ipynb      # Notebook de nettoyage des données
├── pipeline.ipynb              # Notebook du pipeline complet
├── requirements.txt            # Dépendances Python
├── lancer_app.bat              # Raccourci Windows pour lancer l'app
├── rapport-streamlit.html      # Rapport PFE complet
│
├── DATA TOPNET/
│   ├── topnet_all.csv          # Dataset brut fusionné (2 596 textes)
│   ├── topnet_all_clean.csv    # Dataset nettoyé (2 581 textes)
│   └── topnet_all_predictions.csv  # Prédictions MARBERT
│
├── data/
│   ├── train.csv               # Données d'entraînement
│   ├── validation.csv          # Données de validation
│   └── test.csv                # Données de test
│
├── outputs/
│   └── nettoyage_avance_stats.png
│
└── models/                     # ⚠️ Non versionné (621 MB)
    └── MARBERT-balanced-after-gen-neut/
```

> **Note :** Le modèle MARBERT fine-tuné (`models/`) n'est pas inclus dans ce dépôt en raison de sa taille (621 MB). Contacter l'auteur ou télécharger depuis le lien partagé séparément.

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/VOTRE_USERNAME/VOTRE_REPO.git
cd VOTRE_REPO

# 2. Créer un environnement conda (ou venv)
conda create -n sentiment python=3.10
conda activate sentiment

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer l'API Groq
echo "GROQ_API_KEY=votre_cle_groq" > .env

# 5. Placer le modèle MARBERT dans ./models/MARBERT-balanced-after-gen-neut/
```

## Lancer l'application

```bash
streamlit run app.py
```

Ou sous Windows, double-cliquer sur `lancer_app.bat`.

---

## Résultats sur TOPNET Tunisie (2 581 commentaires)

| Sentiment | Nombre | Pourcentage |
|-----------|--------|-------------|
| Positif   | 1 257  | 48.7%       |
| Négatif   |   769  | 29.8%       |
| Neutre    |   555  | 21.5%       |

- **Confiance moyenne** : 93.0%
- **Langues** : arabizi 56% · arabe 27% · français 12% · anglais 5%
- **432 textes** traduits en arabe via Groq avant classification

---

## Technologies utilisées

| Composant | Technologie |
|-----------|-------------|
| Modèle NLP | MARBERT (CAMeL-Lab) fine-tuné |
| Traduction | Groq API — LLaMA 3.3 70B |
| Explicabilité | LIME (Local Interpretable Model-agnostic Explanations) |
| Dashboard | Streamlit + Plotly |
| Détection langue | langdetect + règles Arabizi |

---

## Auteur

Projet réalisé dans le cadre du PFE 2025–2026.
