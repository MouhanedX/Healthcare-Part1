"""Chargement automatique des données locales d'un hôpital.

Chaque client FL ne lit QUE le dossier de son propre hôpital — aucune
donnée ne quitte l'établissement (seuls les poids du modèle circulent).

Le répertoire de base est passé explicitement (run-config `data-dir`,
variable d'environnement `FL_DATA_DIR`, ou défaut du dépôt) car le code
s'exécute depuis le répertoire du FAB installé par Flower, pas depuis
le dépôt.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from src import config


def resolve_base_dir(data_dir: str | None = None) -> Path:
    """Ordre de priorité : argument > variable d'env > défaut du dépôt."""
    if data_dir:
        return Path(data_dir)
    env = os.environ.get("FL_DATA_DIR")
    if env:
        return Path(env)
    return config.HOSPITAL_DATA_DIR


def load_hospital_data(
    hospital: str, data_dir: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Charge (X, y) d'un hôpital depuis son dossier local."""
    csv_path = resolve_base_dir(data_dir) / hospital / config.PATIENTS_CSV
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :-1], data[:, -1]


def load_test_data(data_dir: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Jeu de test global, utilisé uniquement par le serveur (éval. centralisée)."""
    csv_path = resolve_base_dir(data_dir) / config.TEST_CSV
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :-1], data[:, -1]


def load_preprocess_constants(data_dir: str | None = None) -> dict:
    path = resolve_base_dir(data_dir) / config.PREPROCESS_JSON
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def input_dim(data_dir: str | None = None) -> int:
    return len(load_preprocess_constants(data_dir)["features"])
