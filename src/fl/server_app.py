"""Serveur FL (Flower ServerApp) — coordination des 3 hôpitaux.

Stratégie : FedAvg (agrégation pondérée des poids).
À chaque round, le serveur :
  - distribue les poids globaux aux hôpitaux connectés ;
  - agrège les mises à jour (FedAvg) ;
  - évalue le modèle global sur un jeu de test centralisé ;
  - journalise accuracy / loss / numéro de round dans results/runs/<run_id>/metrics.json
    (fichier lu par l'API FastAPI et le futur tableau de bord).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from flwr.common import (
    Context,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from src.data.loader import input_dim, load_test_data
from src.fl.model import build_model

# --------------------------------------------------------------------------- #
#  Journalisation des métriques                                               #
# --------------------------------------------------------------------------- #


class FedAvgLogging(FedAvg):
    """FedAvg qui journalise accuracy/loss par round dans un fichier JSON."""

    def __init__(self, *args, metrics_path: Path, total_rounds: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_path = metrics_path
        self.total_rounds = total_rounds
        self.history: dict = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "total_rounds": total_rounds,
            "rounds": [],
        }
        self._flush()

    def _flush(self) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def _round_entry(self, server_round: int) -> dict:
        for entry in self.history["rounds"]:
            if entry["round"] == server_round:
                return entry
        entry = {"round": server_round}
        self.history["rounds"].append(entry)
        return entry

    @staticmethod
    def _weighted(results: list, key: str) -> float | None:
        # results : liste de (client_proxy, FitRes|EvaluateRes)
        pairs = [
            (res.num_examples, res.metrics)
            for _, res in results
            if key in res.metrics
        ]
        total = sum(n for n, _ in pairs)
        if total == 0:
            return None
        return sum(float(m[key]) * n for n, m in pairs) / total

    def aggregate_fit(self, server_round, results, failures):
        fit_results = [res for _, res in results]
        entry = self._round_entry(server_round)
        entry["train"] = {
            "num_clients": len(results),
            "num_examples": sum(res.num_examples for res in fit_results),
            "accuracy": self._weighted(results, "accuracy"),
            "loss": self._weighted(results, "loss"),
        }
        aggregated = super().aggregate_fit(server_round, results, failures)
        self._flush()
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        entry = self._round_entry(server_round)
        entry["federated_eval"] = {
            "num_clients": len(results),
            "accuracy": self._weighted(results, "accuracy"),
            "loss": float(aggregated[0]) if aggregated else None,
        }
        self._flush()
        return aggregated

    def evaluate(self, server_round, parameters):
        result = super().evaluate(server_round, parameters)
        if result is not None:
            loss, metrics = result
            entry = self._round_entry(server_round)
            entry["centralized_eval"] = {
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "loss": float(loss),
            }
            entry["timestamp"] = datetime.now(timezone.utc).isoformat()
            if server_round >= self.total_rounds:
                self.history["finished_at"] = datetime.now(timezone.utc).isoformat()
                # Poids finaux du modèle global, sauvegardés pour l'inférence
                # (endpoint POST /predict de l'API). Ordre des couches préservé.
                weights_path = self.metrics_path.parent / "weights.npy"
                np.save(
                    weights_path,
                    np.array(parameters_to_ndarrays(parameters), dtype=object),
                    allow_pickle=True,
                )
                print(f"[serveur] poids finaux sauvegardés -> {weights_path}")
            self._flush()
        return result


# --------------------------------------------------------------------------- #
#  Évaluation centralisée (jeu de test global, côté serveur uniquement)       #
# --------------------------------------------------------------------------- #


def get_evaluate_fn(data_dir: str | None = None):
    X_test, y_test = load_test_data(data_dir)
    model = build_model(X_test.shape[1])

    def evaluate(
        server_round: int, parameters: list, config: dict[str, Scalar]
    ) -> tuple[float, dict[str, Scalar]] | None:
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        if server_round == 1 or server_round % 1 == 0:
            print(
                f"[serveur] round {server_round} — évaluation centralisée : "
                f"accuracy={accuracy:.4f}, loss={loss:.4f}"
            )
        return float(loss), {"accuracy": float(accuracy)}

    return evaluate


# --------------------------------------------------------------------------- #
#  ServerApp                                                                  #
# --------------------------------------------------------------------------- #


def server_fn(context: Context) -> ServerAppComponents:
    run_config = context.run_config
    num_rounds = int(run_config.get("num-rounds", 5))
    local_epochs = int(run_config.get("local-epochs", 2))
    batch_size = int(run_config.get("batch-size", 128))
    min_clients = int(run_config.get("min-clients", 3))
    data_dir = run_config.get("data-dir")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    metrics_dir = os.environ.get("FL_RESULTS_DIR") or run_config.get(
        "metrics-dir", "results/runs"
    )
    metrics_path = Path(metrics_dir) / f"run_{run_id}" / "metrics.json"

    print(
        f"[serveur] démarrage — rounds={num_rounds}, min_clients={min_clients}, "
        f"data_dir={data_dir or 'FL_DATA_DIR/défaut'}, métriques -> {metrics_path}"
    )

    strategy = FedAvgLogging(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_fn=get_evaluate_fn(data_dir),
        initial_parameters=ndarrays_to_parameters(
            build_model(input_dim(data_dir)).get_weights()
        ),
        on_fit_config_fn=lambda _round: {
            "local-epochs": local_epochs,
            "batch-size": batch_size,
        },
        metrics_path=metrics_path,
        total_rounds=num_rounds,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
