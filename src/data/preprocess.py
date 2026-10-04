"""Encodage d'un patient individuel pour l'inférence (POST /predict).

Réutilise exactement le pré-traitement de l'entraînement (mêmes catégories
one-hot, mêmes moyennes/écarts-types de standardisation figés dans
preprocess.json) — sinon les prédictions n'auraient aucun sens.
"""

from __future__ import annotations

from src.data.prepare import (
    BINARY_FEATURES,
    GENDER_CATEGORIES,
    NUMERIC_FEATURES,
    SMOKING_CATEGORIES,
)


def encode_patient(patient: dict, constants: dict) -> list[float]:
    """Transforme un patient brut (dict) en vecteur de 15 features.

    `patient` doit contenir : age, bmi, HbA1c_level, blood_glucose_level,
    hypertension, heart_disease, gender, smoking_history.
    """
    row = [
        (float(patient[feat]) - constants["numeric_means"][feat])
        / constants["numeric_stds"][feat]
        for feat in NUMERIC_FEATURES
    ]
    row += [float(patient[feat]) for feat in BINARY_FEATURES]
    row += [1.0 if patient["gender"] == g else 0.0 for g in GENDER_CATEGORIES]
    row += [
        1.0 if patient["smoking_history"] == s else 0.0
        for s in SMOKING_CATEGORIES
    ]
    return row
