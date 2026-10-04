# Smart Healthcare Data Space

Plateforme fédérée permettant à **trois hôpitaux** de co-entraîner un modèle de
**prédiction du risque de diabète** sans jamais partager leurs données
patients : chaque établissement conserve ses données en local, seuls les
**poids du modèle** circulent, agrégés par **FedAvg** via
[Flower](https://flower.ai). Une **API FastAPI** pilote les entraînements,
expose les métriques et sert les prédictions.

---

## 1. Architecture

```
        DONNÉES (ne quittent jamais l'hôpital)          MODÈLE & COORDINATION
┌────────────────────┐                        ┌─────────────────────────────┐
│  fl-hospital-1     │                        │         fl-server           │
│  33 442 patients   │  ① poids globaux ──▶   │   Flower SuperLink          │
│  Hôpital Central   │◀────────────────────   │   + ServerApp (FedAvg)      │
├────────────────────┤                        │                             │
│  fl-hospital-2     │  ② entraînement local  │   ④ agrégation pondérée     │
│  25 963 patients   │  ③ poids mis à jour ─▶ │   ⑤ évaluation test set     │
├────────────────────┤                        │   ⑥ journalisation par round│
│  fl-hospital-3     │  même cycle            └──────────────┬──────────────┘
│  20 595 patients   │                                       │
└────────────────────┘                        results/runs/<id>/metrics.json
                                                        │ weights.npy
                                            ┌───────────▼───────────────┐
                                            │      fl-api (FastAPI)     │
                                            │  :8001  /docs (Swagger)   │
                                            └───────────────────────────┘
```

Chaque composant tourne dans **son propre conteneur Docker** (voir
`docker/docker-compose.yml`) :

| Service | Rôle |
|---|---|
| `fl-server` | Serveur Flower (SuperLink) : orchestre les rounds, exécute la stratégie FedAvg, évalue le modèle global sur le test set, journalise accuracy/loss/round |
| `fl-hospital-1..3` | Un client Flower (SuperNode) par hôpital : monte **uniquement** son dossier de données (lecture seule), entraîne localement |
| `fl-api` | API REST : démarre/suivi des entraînements, historique, statistiques, prédiction |

## 2. Technologies et versions

| Technologie | Version | Rôle |
|---|---|---|
| Python | 3.11 | Langage principal |
| [Flower](https://flower.ai) | 1.36.0 | Framework de Federated Learning (serveur, clients, agrégation FedAvg) |
| TensorFlow / Keras | 2.21.0 / 3.15 | Définition et entraînement du modèle |
| FastAPI | 0.138.2 | API REST + documentation Swagger automatique |
| Uvicorn | 0.49.0 | Serveur ASGI pour l'API |
| NumPy | 2.4.6 | Manipulation des données et des poids |
| Docker / Compose | ≥ 24 / v2 | Conteneurisation et orchestration locale |
| Git | — | Versionnement |

> Versions exactes dans `requirements.txt`. Un venv Python ≥ 3.11 ou Docker
> fonctionnel suffisent.

## 3. Structure du dépôt

```
├── data/
│   ├── raw/
│   │   └── diabetes_prediction_dataset.csv   # dataset source (versionné)
│   └── hospitals/                            # GÉNÉRÉ — non versionné
│       ├── hospital_1..3/patients.csv        #   un CSV par hôpital
│       ├── test_global.csv                   #   jeu de test (serveur)
│       ├── summary.json                      #   stats (lues par l'API)
│       └── preprocess.json                   #   constantes de pré-traitement
├── src/
│   ├── config.py                             # chemins et constantes
│   ├── data/
│   │   ├── prepare.py                        # découpage non-IID en 3 hôpitaux
│   │   ├── loader.py                         # chargement local d'un hôpital
│   │   └── preprocess.py                     # encodage d'un patient (inférence)
│   ├── fl/
│   │   ├── model.py                          # modèle Keras (MLP)
│   │   ├── client_app.py                     # ClientApp Flower (un client = un hôpital)
│   │   └── server_app.py                     # ServerApp Flower (FedAvg + métriques + poids)
│   └── api/
│       └── main.py                           # API FastAPI
├── docker/
│   ├── Dockerfile.fl                         # image serveur + hôpitaux
│   ├── Dockerfile.api                        # image API
│   └── docker-compose.yml                    # topologie complète
├── pyproject.toml                            # configuration de l'app Flower
├── requirements.txt
└── README.md
```

## 4. Les données

Source : [Diabetes Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/almiquej/diabetes-prediction-dataset)
(100 000 patients, 9 colonnes : `gender`, `age`, `hypertension`,
`heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`,
`blood_glucose_level`, cible `diabetes`). Une copie est versionnée dans
`data/raw/`.

### Découpage non-IID entre hôpitaux

`src/data/prepare.py` mélange les patients (graine fixe = reproductible),
retire **20 % pour le jeu de test global** (5 000 patients, utilisés
uniquement par le serveur) et répartit les 80 000 restants par **tirage
probabiliste pondéré** patient par patient, pour reproduire des profils
d'établissements réels :

| Hôpital | Patients | Prévalence diabète | Profil simulé |
|---|---|---|---|
| hospital_1 — Hôpital Central de Tunis | 33 442 | 3,4 % | population générale |
| hospital_2 — Clinique Méditerranée | 25 963 | 7,9 % | population plus âgée |
| hospital_3 — Centre de Diabétologie de Sfax | 20 595 | 17,5 % | recrute les cas diabétiques |

Cette distribution volontairement **non-IID** (chaque hôpital voit une
population différente) est le cas difficile — et réaliste — du Federated
Learning.

### Pré-traitement

Identique pour tous, avec des **constantes partagées figées** dans
`preprocess.json` (indispensable : les poids agrégés n'ont de sens que si les
entrées sont encodées pareillement partout) :

- one-hot de `gender` (3 modalités) et `smoking_history` (6 modalités) ;
- standardisation (moyenne/écart-type) de `age`, `bmi`, `HbA1c_level`,
  `blood_glucose_level` ;
- `hypertension`, `heart_disease` déjà binaires.

Résultat : **15 features** d'entrée.

## 5. Le modèle

MLP Keras compact (~2 050 paramètres), adapté au FL où les poids transitent à
chaque round :

```
Entrée (15) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, sigmoïde)
```

- **Sortie** : probabilité de diabète (seuil 0,5).
- **Loss** : binary cross-entropy ; **optimizer** : Adam (lr 1e-3).
- **Entraînement local** : `batch-size = 128` (≈ 261 mises à jour de poids par
  epoch pour l'hôpital 1), `local-epochs = 2`, paramétrables via l'API.

## 6. Le Federated Learning en pratique

Un **round** = un cycle complet :

1. le serveur diffuse les **poids globaux** aux 3 hôpitaux connectés ;
2. chaque hôpital entraîne la copie reçue **sur ses propres patients**
   (les données ne quittent jamais son conteneur) ;
3. chaque hôpital renvoie ses **poids mis à jour** + son nombre de patients ;
4. le serveur agrège par **FedAvg** — moyenne pondérée par la taille :
   `W = (n₁·W₁ + n₂·W₂ + n₃·W₃) / (n₁+n₂+n₃)` ;
5. il évalue le modèle agrégé sur le test set (évaluation centralisée) ;
6. il journalise tout dans `results/runs/<run_id>/metrics.json`.

Résultat obtenu lors du premier entraînement validé (2 rounds, 1 epoch local,
batch 256 — reproductible) :

| Round | Accuracy globale (test set) | Loss | Accuracy d'entraînement (moy. pondérée) |
|---|---|---|---|
| 0 (poids initiaux) | 68,9 % | 0,625 | — |
| 1 | 92,7 % | 0,176 | 91,2 % |
| 2 | **96,1 %** | 0,113 | 95,3 % |

Contenu de `metrics.json` par round :

- `train` : accuracy/loss **d'entraînement** agrégées côté serveur (ce que les
  hôpitaux rapportent) ;
- `federated_eval` : modèle global évalué par chaque hôpital sur ses données
  puis agrégé ;
- `centralized_eval` : modèle global évalué par le serveur sur le **test set**
  → chiffre de référence ;
- à la fin du run, les **poids finaux** sont sauvegardés dans
  `results/runs/<run_id>/weights.npy` (servent à `/predict`).

## 7. Mise en route

### Prérequis

- Docker Desktop (avec Compose v2) — **ou** Python ≥ 3.11 pour le mode local ;
- aucune clé API, aucun service externe.

### 1. Préparer les données (une seule fois)

```bash
python -m src.data.prepare        # (avec le venv du projet activé)
```

Génère `data/hospitals/` (3 CSV + test + constantes) et affiche les stats.

### 2. Lancer la plateforme (Docker, recommandé)

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

Vérifications :

```bash
docker compose -f docker/docker-compose.yml ps   # 5 conteneurs "Up"
curl http://localhost:8001/health                # {"status": "ok", ...}
```

### 3. Lancer un entraînement fédéré

Via Swagger : <http://localhost:8001/docs> → `POST /train/start` → *Try it
out* → corps :

```json
{"num_rounds": 3, "local_epochs": 2, "batch_size": 128}
```

ou en ligne de commande :

```bash
curl -X POST http://localhost:8001/train/start \
     -H "Content-Type: application/json" \
     -d '{"num_rounds": 3, "local_epochs": 2, "batch_size": 128}'
```

Suivre en direct :

```bash
curl http://localhost:8001/train/status    # round courant, accuracy, loss
docker logs -f fl-hospital-1               # "fit terminé — 33442 patients..."
```

### 4. Consulter les résultats et prédire

```bash
curl http://localhost:8001/train/results   # historique des rounds + best_accuracy
curl http://localhost:8001/runs            # tous les runs

curl -X POST http://localhost:8001/predict \
     -H "Content-Type: application/json" \
     -d '{"age": 65, "gender": "Male", "hypertension": true,
          "heart_disease": true, "smoking_history": "former",
          "bmi": 34.5, "HbA1c_level": 8.2, "blood_glucose_level": 240}'
# -> {"probability_diabetes": 0.9377, "prediction": "risque de diabète",
#     "risk_level": "élevé", ...}
```

`/predict` charge les poids du **dernier run terminé** — il faut donc avoir
lancé au moins un entraînement au préalable.

### Mode local sans Docker (debug)

Quatre terminaux, venv activé :

```bash
# 1. serveur Flower
flower-superlink --insecure

# 2-4. un client par hôpital (Windows : set VAR=valeur ; Linux : export)
HOSPITAL_ID=0 FL_DATA_DIR=<chemin absolu>/data/hospitals \
  flower-supernode --insecure --superlink 127.0.0.1:9092 --port 19094
HOSPITAL_ID=1 FL_DATA_DIR=<chemin absolu>/data/hospitals \
  flower-supernode --insecure --superlink 127.0.0.1:9092 --port 29094
HOSPITAL_ID=2 FL_DATA_DIR=<chemin absolu>/data/hospitals \
  flower-supernode --insecure --superlink 127.0.0.1:9092 --port 39094
```

Puis déclarer la connexion dans `~/.flwr/config.toml` et soumettre :

```toml
[superlink.localreal]
address = "127.0.0.1:9093"
insecure = true
```

```bash
flwr run . localreal -c 'num-rounds=2 data-dir="<chemin absolu>/data/hospitals"'
```

> **Piège Flower 1.36** : le mode par défaut de `flwr run` est une simulation
> qui exige `ray`. La procédure ci-dessus (SuperLink + SuperNodes réels) est
> le chemin validé, sans dépendance supplémentaire.

## 8. API — référence

| Méthode | Route | Description |
|---|---|---|
| GET | `/health` | État de l'API |
| GET | `/hospitals` | Statistiques des 3 jeux de données (taille, prévalence) |
| POST | `/train/start` | Lance un entraînement (rounds, epochs, batch, min_clients) |
| GET | `/train/status` | Progression en direct (round courant, accuracy, loss) |
| GET | `/train/results` | Détail du dernier run (`?run=<run_id>`) |
| GET | `/runs` | Historique de tous les runs |
| POST | `/train/stop` | Arrête l'entraînement en cours |
| POST | `/predict` | Prédiction pour un patient (dernier modèle global) |

Documentation interactive complète : <http://localhost:8001/docs>.

## 9. Confidentialité — ce qui circule et ce qui ne circule pas

- ✅ circulent sur le réseau : **poids du modèle** (~8 Ko), métriques,
  configuration des rounds ;
- ❌ ne circulent **jamais** : données patients. Chaque conteneur hôpital monte
  uniquement son propre dossier en **lecture seule** (`docker-compose.yml`),
  le serveur ne voit que le test set, et l'API ne touche à aucune donnée
  hospitalière.

À connaître pour la suite : le FL protège les données brutes, pas
nécessairement ce que peuvent révéler les poids eux-mêmes (attaques par
inversion, fuites par gradient) — d'où les pistes chiffrement/differential
privacy listées ci-dessous.

## 10. Prochaines étapes

1. **Connecteurs EDC** (Eclipse Dataspace Components) : publier/découvrir des
   datasets, négocier des contrats d'échange, journaliser les échanges entre
   les établissements ;
2. **Kubernetes** (Minikube/Kind) : porter la topologie docker-compose en
   pods/Deployments/Services ;
3. **Tableau de bord** : consommer `/train/status`, `/runs` et `/hospitals`
   pour afficher hôpitaux connectés, progression, accuracy, temps d'exécution ;
4. **Sécurité** : chiffrement des communications (TLS au lieu de `--insecure`),
   differential privacy / secure aggregation sur les poids.
