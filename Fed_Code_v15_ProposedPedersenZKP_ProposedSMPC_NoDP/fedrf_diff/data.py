"""Heart-disease data: loading, cleaning and the 4-hospital federated split.

The CSV has no source column, but its rows are ordered by source dataset:

    rows    0-293   Hungary          (294)
    rows  294-416   Switzerland      (123)
    rows  417-616   VA Long Beach    (200)
    rows  617-886   Statlog          (270)  same patients as Cleveland
    rows  887-1189  Cleveland        (303)

Statlog is a subset of the Cleveland Clinic patients, so it is merged into the
Cleveland client. Duplicate rows are kept for training (as requested); each
row gets a ``patient_id`` shared by its copies so splits can respect it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

NUMERIC = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]
BINARY = ["sex", "fasting blood sugar", "exercise angina"]
NOMINAL = ["chest pain type", "resting ecg", "ST slope"]
FEATURES = [
    "age", "sex", "chest pain type", "resting bp s", "cholesterol",
    "fasting blood sugar", "resting ecg", "max heart rate", "exercise angina",
    "oldpeak", "ST slope",
]
TARGET = "target"

# (client name, first row, end row) in the original file.
HOSPITAL_BLOCKS = (
    ("Client 1(Hungary)", 0, 294),
    ("Client 2(Switzerland)", 294, 417),
    ("Client 3(VA Long Beach)", 417, 617),
    ("Client 4(Cleveland)", 617, 1190),  # Statlog 617-886 + Cleveland 887-1189
)
CLIENT_NAMES = tuple(name for name, _, _ in HOSPITAL_BLOCKS)


def numeric_mask() -> np.ndarray:
    """Boolean mask over FEATURES: True for numeric, False for binary/nominal."""
    return np.array([f in NUMERIC for f in FEATURES])


def load_heart_data(path: str, drop_invalid_rows: bool = True) -> pd.DataFrame:
    """Load the CSV and add ``hospital`` and ``patient_id`` columns."""
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    if len(df) != HOSPITAL_BLOCKS[-1][2]:
        raise ValueError(
            f"expected {HOSPITAL_BLOCKS[-1][2]} rows (row order encodes the hospital), "
            f"got {len(df)}"
        )
    hospital = np.empty(len(df), dtype=object)
    for name, start, end in HOSPITAL_BLOCKS:
        hospital[start:end] = name
    df["hospital"] = hospital
    df["patient_id"] = df.groupby(FEATURES + [TARGET], sort=False).ngroup()
    if drop_invalid_rows:
        invalid = (df["resting bp s"] == 0) | (df["ST slope"] == 0)
        df = df.loc[~invalid]
    return df.reset_index(drop=True)


@dataclass
class ClientData:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    train_patient_ids: np.ndarray = field(repr=False)
    val_patient_ids: np.ndarray = field(repr=False)

    @property
    def n_train(self) -> int:
        return len(self.y_train)


@dataclass
class FederatedData:
    clients: list[ClientData]
    X_landmarks: np.ndarray
    landmark_patient_ids: np.ndarray


def _sample_landmarks(df: pd.DataFrame, n_landmarks: int, rng: np.random.Generator) -> np.ndarray:
    """Pick ``n_landmarks`` unique patients, allocated to hospitals by size."""
    unique = df.drop_duplicates("patient_id")
    sizes = unique.groupby("hospital", sort=False).size().reindex(CLIENT_NAMES)
    quota = np.floor(sizes / sizes.sum() * n_landmarks).astype(int)
    # Hand out the remainder to the largest fractional parts.
    remainder = n_landmarks - quota.sum()
    frac = (sizes / sizes.sum() * n_landmarks) - quota
    for name in frac.sort_values(ascending=False).index[:remainder]:
        quota[name] += 1
    chosen = []
    for name in CLIENT_NAMES:
        ids = unique.loc[unique["hospital"] == name, "patient_id"].to_numpy()
        chosen.extend(rng.choice(ids, size=quota[name], replace=False))
    return np.array(sorted(chosen))


def make_federated_split(
    df: pd.DataFrame,
    n_landmarks: int,
    val_fraction: float,
    group_aware: bool,
    seed: int,
) -> FederatedData:
    """Split into the public landmark cohort and 4 hospital clients."""
    rng = np.random.default_rng(seed)
    landmark_ids = _sample_landmarks(df, n_landmarks, rng) if n_landmarks > 0 else np.array([], int)
    is_landmark = df["patient_id"].isin(landmark_ids)
    landmarks = df.loc[is_landmark].drop_duplicates("patient_id").sort_values("patient_id")
    rest = df.loc[~is_landmark]

    clients = []
    for name in CLIENT_NAMES:
        part = rest.loc[rest["hospital"] == name]
        X = part[FEATURES].to_numpy(dtype=float)
        y = part[TARGET].to_numpy(dtype=int)
        pid = part["patient_id"].to_numpy()
        if group_aware:
            n_splits = max(2, round(1 / val_fraction))
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            train_idx, val_idx = next(splitter.split(X, y, groups=pid))
        else:
            train_idx, val_idx = train_test_split(
                np.arange(len(y)), test_size=val_fraction, stratify=y, random_state=seed
            )
        clients.append(
            ClientData(
                name=name,
                X_train=X[train_idx], y_train=y[train_idx],
                X_val=X[val_idx], y_val=y[val_idx],
                train_patient_ids=pid[train_idx], val_patient_ids=pid[val_idx],
            )
        )
    return FederatedData(
        clients=clients,
        X_landmarks=landmarks[FEATURES].to_numpy(dtype=float),
        landmark_patient_ids=landmarks["patient_id"].to_numpy(),
    )
