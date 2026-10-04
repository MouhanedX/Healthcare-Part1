"""Modèle de prédiction du diabète (Keras MLP).

Architecture volontairement légère : en FL, le modèle est transmis
allez-retour entre serveur et clients à chaque round.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silence les logs TF

import keras
from keras import layers


def build_model(input_dim: int, seed: int = 42) -> keras.Model:
    keras.utils.set_random_seed(seed)
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model
