"""Module 1 — Préparation des datasets des trois hôpitaux.

Lit le CSV Kaggle brut (diabetes_prediction_dataset.csv), applique un
pré-traitement déterministe, puis découpe les données en trois jeux
locaux « non-IID » (un par hôpital) + un jeu de test global pour
l'évaluation centralisée côté serveur.

Les données de chaque hôpital sont écrites dans son propre dossier :
    data/hospitals/hospital_1/patients.csv
    data/hospitals/hospital_2/patients.csv
    data/hospitals/hospital_3/patients.csv
    data/hospitals/test_global.csv
    data/hospitals/summary.json      (stats lues par l'API)

Chaque hôpital garde une « personnalité » différente (taille, prévalence
du diabète, âge moyen) pour obtenir un entraînement réellement non-IID.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from src import config

# -----------------------------------------------------------------------------
# Pré-traitement (déterministe, appliqué de manière identique à tous les hôpitaux)
# -----------------------------------------------------------------------------

GENDER_CATEGORIES = ["Female", "Male", "Other"]
SMOKING_CATEGORIES = ["never", "current", "former", "ever", "not current", "No Info"]

NUMERIC_FEATURES = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
BINARY_FEATURES = ["hypertension", "heart_disease"]

FEATURES = (
    NUMERIC_FEATURES + BINARY_FEATURES
    + [f"gender_{g}" for g in GENDER_CATEGORIES]
    + [f"smoking_{s}" for s in SMOKING_CATEGORIES]
)
TARGET = "diabetes"


def _raw_csv_path() -> Path:
    for candidate in config.RAW_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "CSV brut introuvable. Placez diabetes_prediction_dataset.csv dans "
        "data/raw/ ou Dataset/."
    )


def _read_raw(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _preprocess(rows: list[dict]) -> tuple[list[list[float]], list[int], dict]:
    """Encodage one-hot + standardisation. Retourne (X, y, constantes)."""
    means, stds = {}, {}
    for feat in NUMERIC_FEATURES:
        values = [float(r[feat]) for r in rows]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        means[feat], stds[feat] = mean, var ** 0.5 or 1.0

    X: list[list[float]] = []
    y: list[int] = []
    for r in rows:
        row = [
            (float(r[feat]) - means[feat]) / stds[feat] for feat in NUMERIC_FEATURES
        ]
        row += [float(r[feat]) for feat in BINARY_FEATURES]
        row += [1.0 if r["gender"] == g else 0.0 for g in GENDER_CATEGORIES]
        row += [1.0 if r["smoking_history"] == s else 0.0 for s in SMOKING_CATEGORIES]
        X.append(row)
        y.append(int(r[TARGET]))

    constants = {
        "features": FEATURES,
        "target": TARGET,
        "numeric_means": means,
        "numeric_stds": stds,
        "gender_categories": GENDER_CATEGORIES,
        "smoking_categories": SMOKING_CATEGORIES,
    }
    return X, y, constants


def _assign_hospitals(
    X: list[list[float]], y: list[int], rng: random.Random
) -> list[int]:
    """Attribution non-IID des patients aux 3 hôpitaux.

    hospital_1 : hôpital généraliste (population globale).
    hospital_2 : clinique avec une population plus âgée.
    hospital_3 : centre de diabétologie (prévalence de diabète plus élevée).
    """
    assignment: list[int] = []
    for row, label in zip(X, y):
        age = row[0]  # age standardisée : >0 ≈ au-dessus de la moyenne (~41 ans)
        w = [1.0, 0.55, 0.45]
        if label == 1:  # le centre spécialisé attire beaucoup de cas diabétiques
            w = [0.5, 0.35, 2.6]
        if age > 0.8:  # patients âgés plutôt orientés vers la clinique
            w = [0.8, 1.8, 0.9]
        if age > 1.5 and label == 1:
            w = [0.4, 1.2, 2.4]
        total = sum(w)
        pick = rng.random() * total
        acc = 0.0
        for i, weight in enumerate(w):
            acc += weight
            if pick <= acc:
                assignment.append(i)
                break
        else:
            assignment.append(2)
    return assignment


def _write_csv(path: Path, rows: list[list[float]], labels: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURES + [TARGET])
        for row, label in zip(rows, labels):
            writer.writerow([f"{v:.6f}" for v in row] + [label])


def main() -> None:
    raw_path = _raw_csv_path()
    print(f"Lecture du dataset brut : {raw_path}")
    rows = _read_raw(raw_path)
    print(f"Patients bruts : {len(rows)}")

    rng = random.Random(config.SEED)
    rng.shuffle(rows)

    X, y, constants = _preprocess(rows)

    # --- Jeu de test global (évaluation centralisée côté serveur) : 20 % -----
    split = int(len(X) * 0.2)
    test_idx = list(range(0, split, 4))  # échantillon stratifié simple (1/4 des 20%)
    train_idx = [i for i in range(len(X)) if i >= split]

    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]

    # --- Découpage non-IID entre les trois hôpitaux --------------------------
    assignment = _assign_hospitals(X_train, y_train, rng)

    buckets: dict[int, tuple[list[list[float]], list[int]]] = {i: ([], []) for i in range(3)}
    for i, hosp in enumerate(assignment):
        buckets[hosp][0].append(X_train[i])
        buckets[hosp][1].append(y_train[i])

    out_dir = config.HOSPITAL_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / config.PREPROCESS_JSON).write_text(
        json.dumps(constants, indent=2), encoding="utf-8"
    )
    _write_csv(out_dir / config.TEST_CSV, X_test, y_test)

    summary = {
        "dataset": "Diabetes Prediction Dataset (Kaggle, 100k patients)",
        "total_patients_raw": len(rows),
        "test_global": {"num_examples": len(y_test), "prevalence": sum(y_test) / len(y_test)},
        "hospitals": {},
    }
    for i in range(3):
        hospital = config.HOSPITALS[i]
        X_h, y_h = buckets[i]
        _write_csv(out_dir / hospital / config.PATIENTS_CSV, X_h, y_h)
        summary["hospitals"][hospital] = {
            "name": config.HOSPITAL_NAMES[hospital],
            "num_examples": len(y_h),
            "prevalence": round(sum(y_h) / len(y_h), 4),
            "data_file": str(out_dir / hospital / config.PATIENTS_CSV),
        }
        print(
            f"{hospital} ({config.HOSPITAL_NAMES[hospital]}): "
            f"{len(y_h)} patients, prévalence diabète {sum(y_h)/len(y_h):.2%}"
        )
    print(
        f"Test global : {len(y_test)} patients, "
        f"prévalence {sum(y_test)/len(y_test):.2%}"
    )

    (out_dir / config.SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Datasets écrits dans {out_dir}")


if __name__ == "__main__":
    main()
