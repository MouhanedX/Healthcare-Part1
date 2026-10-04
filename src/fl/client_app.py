"""Client FL d'un hôpital (Flower ClientApp).

Chaque supernode représente un hôpital. Le client :
  1. charge UNIQUEMENT les données de son hôpital (stockage local) ;
  2. entraîne le modèle reçu du serveur sur ses données ;
  3. renvoie les poids mis à jour + métriques locales (accuracy, loss).
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays

from src import config
from src.data.loader import load_hospital_data
from src.fl.model import build_model

# Détermination de l'hôpital :
#   - en Docker : variable d'environnement HOSPITAL_ID (0, 1 ou 2) ;
#   - en local  : partition-id du supernode (0, 1, 2 ...).
_env_hospital = os.environ.get("HOSPITAL_ID")


def _hospital_for(context: Context) -> str:
    if _env_hospital is not None:
        index = int(_env_hospital) % len(config.HOSPITALS)
    else:
        index = int(context.node_config.get("partition-id", 0)) % len(config.HOSPITALS)
    return config.HOSPITALS[index]


class DiabetesClient(NumPyClient):
    def __init__(self, hospital: str, data_dir: str | None = None):
        self.hospital = hospital
        self.X, self.y = load_hospital_data(hospital, data_dir)
        self.model = build_model(self.X.shape[1])

    def fit(self, parameters: NDArrays, config: dict):
        self.model.set_weights(parameters)
        epochs = int(config.get("local-epochs", 2))
        batch_size = int(config.get("batch-size", 128))
        history = self.model.fit(
            self.X,
            self.y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
        num_examples = int(self.X.shape[0])
        metrics = {
            "accuracy": float(history.history["accuracy"][-1]),
            "loss": float(history.history["loss"][-1]),
            "hospital": self.hospital,
        }
        print(
            f"[{self.hospital}] fit terminé — {num_examples} patients, "
            f"accuracy locale={metrics['accuracy']:.4f}, loss={metrics['loss']:.4f}"
        )
        return self.model.get_weights(), num_examples, metrics

    def evaluate(self, parameters: NDArrays, config: dict):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X, self.y, verbose=0)
        num_examples = int(self.X.shape[0])
        metrics = {"accuracy": float(accuracy), "hospital": self.hospital}
        print(
            f"[{self.hospital}] évaluation locale — accuracy={accuracy:.4f}, "
            f"loss={loss:.4f}"
        )
        return float(loss), num_examples, metrics


def client_fn(context: Context) -> DiabetesClient:
    data_dir = context.run_config.get("data-dir")
    return DiabetesClient(_hospital_for(context), data_dir)


app = ClientApp(client_fn=client_fn)
