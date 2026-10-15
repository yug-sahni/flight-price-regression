"""Global configuration. Every source of randomness in the project is seeded from SEED."""
import os
import random
from pathlib import Path

import numpy as np

SEED = 42
N_FOLDS = 5
HOLDOUT_FRAC = 0.20          # untouched until after tuning; gives an unbiased estimate of generalisation

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
SUB = OUT / "submissions"
for p in (FIG, SUB):
    p.mkdir(parents=True, exist_ok=True)

TARGET = "price"
N_JOBS = 2


def seed_everything(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
