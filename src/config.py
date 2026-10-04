"""Configuration globale du projet Smart Healthcare Data Space."""

from __future__ import annotations

import os
from pathlib import Path

# Racine du dépôt (src/config.py -> racine = 2 niveaux au-dessus)
REPO_ROOT = Path(__file__).resolve().parents[1]

# Datasets -------------------------------------------------------------------
# CSV brut (Kaggle) : on accepte plusieurs emplacements
RAW_CANDIDATES = [
    REPO_ROOT / "data" / "raw" / "diabetes_prediction_dataset.csv",
    REPO_ROOT / "Dataset" / "diabetes_prediction_dataset.csv",
]

# Répertoire contenant les données découpées par hôpital (peut être monté
# en volume Docker : chaque conteneur hôpital ne voit QUE son propre dossier).
HOSPITAL_DATA_DIR = Path(os.environ.get("FL_DATA_DIR", REPO_ROOT / "data" / "hospitals"))

HOSPITALS = ["hospital_1", "hospital_2", "hospital_3"]
HOSPITAL_NAMES = {
    "hospital_1": "Hôpital Central de Tunis",
    "hospital_2": "Clinique Méditerranée",
    "hospital_3": "Centre de Diabétologie de Sfax",
}

TEST_CSV = "test_global.csv"
PATIENTS_CSV = "patients.csv"
SUMMARY_JSON = "summary.json"
PREPROCESS_JSON = "preprocess.json"

# Résultats des runs (métriques JSON par round, lus par l'API / le dashboard)
RESULTS_DIR = Path(os.environ.get("FL_RESULTS_DIR", REPO_ROOT / "results" / "runs"))

# Seed global pour la reproductibilité
SEED = 42
