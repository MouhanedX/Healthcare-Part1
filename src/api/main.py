"""API FastAPI du Smart Healthcare Data Space.

Endpoints :
  GET  /health            — état de l'API
  GET  /hospitals         — statistiques des 3 jeux de données hospitaliers
  POST /train/start       — lance un entraînement fédéré (serveur + 3 clients)
  GET  /train/status      — progression en direct (round courant, accuracy…)
  GET  /train/results     — résultat du dernier run (ou ?run=<id>)
  GET  /runs              — historique de tous les entraînements
  POST /train/stop        — arrête l'entraînement en cours
  POST /predict           — prédit le risque de diabète d'un patient avec le
                            dernier modèle global entraîné
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import config
from src.data.loader import load_preprocess_constants
from src.data.preprocess import encode_patient
from src.fl.model import build_model

app = FastAPI(
    title="Smart Healthcare Data Space",
    description=(
        "Plateforme permettant à trois hôpitaux d'entraîner conjointement un "
        "modèle de prédiction du diabète via Federated Learning (Flower + "
        "FedAvg) sans partager leurs données."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FEDERATION = os.environ.get("FL_FEDERATION", "localreal")


class TrainRequest(BaseModel):
    num_rounds: int = Field(default=5, ge=1, le=100, description="Rounds d'agrégation")
    local_epochs: int = Field(default=2, ge=1, le=20, description="Epochs locales par round")
    batch_size: int = Field(default=128, ge=16, le=1024)
    min_clients: int = Field(default=3, ge=1, le=3, description="Hôpitaux minimum requis")


class TrainJob:
    """État de l'entraînement en cours (un seul à la fois)."""

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.started_at: float | None = None
        self.settings: dict = {}
        self.log_path: Path | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, req: TrainRequest) -> None:
        if self.running:
            raise HTTPException(409, "Un entraînement est déjà en cours.")

        flwr_exe = shutil.which("flwr") or str(
            Path(sys.executable).parent / ("flwr.exe" if os.name == "nt" else "flwr")
        )
        if not Path(flwr_exe).exists():
            raise HTTPException(500, f"Executable flwr introuvable : {flwr_exe}")

        data_dir = str(config.HOSPITAL_DATA_DIR).replace("\\", "/")
        metrics_dir = str(config.RESULTS_DIR).replace("\\", "/")
        overrides = (
            f"num-rounds={req.num_rounds} "
            f"local-epochs={req.local_epochs} "
            f"batch-size={req.batch_size} "
            f"min-clients={req.min_clients} "
            f'data-dir="{data_dir}" '
            f'metrics-dir="{metrics_dir}"'
        )
        self.log_path = config.REPO_ROOT / "logs" / f"api_run_{int(time.time())}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [flwr_exe, "run", ".", FEDERATION, "-c", overrides],
            cwd=config.REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        self.started_at = time.time()
        self.settings = req.model_dump()

    def stop(self) -> None:
        if not self.running:
            raise HTTPException(409, "Aucun entraînement en cours.")
        self.process.terminate()
        self.process = None


job = TrainJob()


def _latest_metrics_file() -> Path | None:
    runs = sorted(config.RESULTS_DIR.glob("run_*/metrics.json"))
    return runs[-1] if runs else None


def _read_metrics(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    data["run_id"] = path.parent.name
    return data


# --------------------------------------------------------------------------- #
#  Endpoints                                                                   #
# --------------------------------------------------------------------------- #


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "smarthealth-api",
        "time": datetime.now(timezone.utc).isoformat(),
        "training_running": job.running,
    }


@app.get("/hospitals")
def hospitals() -> dict:
    summary_path = config.HOSPITAL_DATA_DIR / config.SUMMARY_JSON
    if not summary_path.exists():
        raise HTTPException(
            404,
            "Datasets non préparés. Lancez : python -m src.data.prepare",
        )
    with summary_path.open(encoding="utf-8") as f:
        return json.load(f)


@app.post("/train/start")
def train_start(req: TrainRequest) -> dict:
    job.start(req)
    return {
        "message": "Entraînement fédéré démarré.",
        "federation": FEDERATION,
        "settings": job.settings,
        "log_file": str(job.log_path),
    }


@app.get("/train/status")
def train_status() -> dict:
    metrics_file = _latest_metrics_file()
    status: dict = {
        "running": job.running,
        "federation": FEDERATION,
        "started_at": (
            datetime.fromtimestamp(job.started_at, tz=timezone.utc).isoformat()
            if job.started_at
            else None
        ),
        "elapsed_seconds": round(time.time() - job.started_at) if job.started_at else None,
        "settings": job.settings or None,
    }
    if metrics_file:
        metrics = _read_metrics(metrics_file)
        rounds = metrics.get("rounds", [])
        current_round = rounds[-1]["round"] if rounds else 0
        finished = "finished_at" in metrics
        # `flwr run` se détache après soumission : l'entraînement continue
        # côté serveur. On considère l'entraînement actif tant que le run
        # n'est pas terminé et qu'un round a été vu il y a moins de 15 min.
        last_ts = rounds[-1].get("timestamp") if rounds else None
        recent = False
        if last_ts:
            try:
                last_dt = datetime.fromisoformat(last_ts)
                recent = (datetime.now(timezone.utc) - last_dt).total_seconds() < 900
            except ValueError:
                recent = False
        status.update(
            {
                "run_id": metrics["run_id"],
                "current_round": current_round,
                "total_rounds": metrics.get("total_rounds"),
                "finished": finished,
                "latest": rounds[-1] if rounds else None,
            }
        )
        if job.started_at and (job.running or (not finished and recent)):
            status["running"] = True
    return status


@app.get("/train/results")
def train_results(run: str | None = None) -> dict:
    if run:
        metrics_file = config.RESULTS_DIR / run / "metrics.json"
    else:
        metrics_file = _latest_metrics_file()
    if not metrics_file or not metrics_file.exists():
        raise HTTPException(404, "Aucun résultat d'entraînement disponible.")
    metrics = _read_metrics(metrics_file)
    rounds = metrics.get("rounds", [])
    central = [
        {"round": r["round"], **r.get("centralized_eval", {})}
        for r in rounds
        if "centralized_eval" in r
    ]
    metrics["best_accuracy"] = max((c["accuracy"] for c in central), default=None)
    return metrics


@app.get("/runs")
def list_runs() -> dict:
    runs = []
    for metrics_file in sorted(config.RESULTS_DIR.glob("run_*/metrics.json")):
        try:
            metrics = _read_metrics(metrics_file)
            rounds = metrics.get("rounds", [])
            central = [
                r.get("centralized_eval", {}).get("accuracy")
                for r in rounds
                if "centralized_eval" in r
            ]
            runs.append(
                {
                    "run_id": metrics["run_id"],
                    "started_at": metrics.get("started_at"),
                    "total_rounds": metrics.get("total_rounds"),
                    "rounds_done": len(rounds),
                    "finished": "finished_at" in metrics,
                    "best_accuracy": max(central) if central else None,
                }
            )
        except (json.JSONDecodeError, OSError):
            continue
    return {"runs": runs}


@app.post("/train/stop")
def train_stop() -> dict:
    job.stop()
    return {"message": "Entraînement arrêté."}


# --------------------------------------------------------------------------- #
#  Inférence — prédiction du risque de diabète d'un patient                    #
# --------------------------------------------------------------------------- #


class PatientInput(BaseModel):
    """Patient brut (mêmes champs que le CSV Kaggle d'origine)."""

    age: float = Field(..., ge=1, le=120, examples=[58])
    gender: Literal["Female", "Male", "Other"] = Field(..., examples=["Male"])
    hypertension: bool = Field(default=False)
    heart_disease: bool = Field(default=False)
    smoking_history: Literal[
        "never", "current", "former", "ever", "not current", "No Info"
    ] = Field(default="No Info")
    bmi: float = Field(..., ge=10, le=80, examples=[31.5])
    HbA1c_level: float = Field(..., ge=3, le=20, examples=[7.8])
    blood_glucose_level: int = Field(..., ge=40, le=500, examples=[210])


_model_cache: dict = {}


def _load_latest_model() -> tuple[object | None, str | None]:
    """Charge le dernier run disposant de poids finaux (mise en cache)."""
    for run_dir in sorted(config.RESULTS_DIR.glob("run_*"), reverse=True):
        weights_path = run_dir / "weights.npy"
        if weights_path.exists():
            if _model_cache.get("run_id") != run_dir.name:
                constants = load_preprocess_constants()
                model = build_model(len(constants["features"]))
                model.set_weights(list(np.load(weights_path, allow_pickle=True)))
                _model_cache.update(run_id=run_dir.name, model=model)
            return _model_cache["model"], _model_cache["run_id"]
    return None, None


@app.post("/predict")
def predict(patient: PatientInput) -> dict:
    model, run_id = _load_latest_model()
    if model is None:
        raise HTTPException(
            404,
            "Aucun modèle entraîné disponible. Lancez d'abord POST /train/start "
            "et attendez la fin du run (GET /train/status).",
        )
    constants = load_preprocess_constants()
    x = np.array([encode_patient(patient.model_dump(), constants)], dtype=np.float32)
    probability = float(model.predict(x, verbose=0)[0][0])
    return {
        "probability_diabetes": round(probability, 4),
        "prediction": "risque de diabète" if probability >= 0.5 else "non diabétique",
        "risk_level": (
            "élevé" if probability >= 0.7 else "modéré" if probability >= 0.3 else "faible"
        ),
        "model_run_id": run_id,
        "input": patient.model_dump(),
    }
