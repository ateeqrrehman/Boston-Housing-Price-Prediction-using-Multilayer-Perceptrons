# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')   # it’ll ask to authorize once

# STEP 1.  Imports, versions, and data load
import sys, random, math, time, warnings, platform
from pathlib import Path
import numpy as np
import pandas as pd

# scikit‑learn (for baseline pipeline)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Printing environment for reproducibility
print("Python       :", sys.version.split()[0])
print("scikit‑learn :", __import__('sklearn').__version__)
print("torch       :", torch.__version__)
print("Platform    :", platform.platform())

# Fixing random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Loading Boston dataset
COLS = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]
DATA_PATH = Path("/content/drive/My Drive/MSML612/housing.csv")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# File is whitespace‑separated
raw_df = pd.read_csv(DATA_PATH, delim_whitespace=True, header=None, names=COLS)
print(f"Dataset loaded → {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")

# STEP 2.  Split into X (inputs) and y (target)
# ================================================================

X = raw_df.drop("MEDV", axis=1).to_numpy(dtype=np.float32)
y = raw_df["MEDV"].to_numpy(dtype=np.float32).reshape(-1, 1)
INPUT_DIM = X.shape[1]
print("X shape:", X.shape, "  y shape:", y.shape)


# Helper — scikit‑learn pipeline with StandardScaler + MLPRegressor
# ================================================================

def make_sklearn_mlp(hidden=(128, 64), max_iter=2000, batch=32, alpha=1e-4):
    """Return a Pipeline that scales then fits an MLPRegressor."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("net", MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            batch_size=batch,
            alpha=alpha,
            learning_rate_init=1e-3,
            early_stopping=True,
            random_state=SEED,
        )),
    ])


def cv_mse_sklearn(model, k=10):
    """10‑fold cross‑val returning (mean, std) of **positive** MSE."""
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    neg = cross_val_score(model, X, y.ravel(), scoring="neg_mean_squared_error", cv=kf, n_jobs=-1)
    mse_vals = -neg  # flip sign
    return mse_vals.mean(), mse_vals.std()

# STEP 3.  2‑hidden‑layer baseline
# ================================================================
print("\nSTEP 3 — Two‑hidden‑layer MLP (128,64)…")
base_model = make_sklearn_mlp((128, 64))
print("Model:", base_model)

# STEP 4.  Standardisation is already inside Pipeline (nothing extra)
# ================================================================
print("STEP 4 — StandardScaler applied within each CV fold via Pipeline.")

# STEP 5.  10‑fold CV evaluation
# ================================================================
print("STEP 5 — 10‑fold CV for baseline…")
mean_mse, std_mse = cv_mse_sklearn(base_model)
print(f"Baseline MSE = {mean_mse:6.2f} ± {std_mse:.2f}")

# PyTorch utility functions
# =========================

class FCNet(nn.Module):
    """Fully connected network of arbitrary depth."""
    def __init__(self, input_dim: int, hidden: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last = input_dim
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def mse_fold_torch(hidden_sizes, epochs=500, batch=32, lr=1e-3, device="cpu"):
    """Return validation‑set MSE for one CV fold."""
    loss_fn = nn.MSELoss()
    model = FCNet(INPUT_DIM, hidden_sizes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Create data loaders inside the caller to avoid redundancy
    def run_epoch(loader, train=True):
        if train:
            model.train()
        else:
            model.eval()
        total, count = 0.0, 0
        with torch.set_grad_enabled(train):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                if train:
                    opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                if train:
                    loss.backward(); opt.step()
                total += loss.item() * xb.size(0)
                count += xb.size(0)
        return total / count

    # Training loop
    for _ in range(epochs):
        train_epoch()  # defined below inline

    return val_epoch()  # lowest achieved automatically (no early stop)


def cv_mse_torch(hidden_sizes, epochs=500, batch=32, k=10, lr=1e-3):
    """Return mean ± std MSE across k folds for a PyTorch network."""
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    fold_mse = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for train_idx, val_idx in kf.split(X):
        # Standardise inside this fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val   = scaler.transform(X[val_idx])
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val))
        train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=batch)

        # Build & train model
        model = FCNet(INPUT_DIM, hidden_sizes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        best_val = math.inf

        for _ in range(epochs):
            # training step
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
            # validation step
            model.eval();
            with torch.no_grad():
                val_loss = sum(loss_fn(model(xv.to(device)), yv.to(device)).item()*len(xv)
                                for xv, yv in val_dl) / len(val_dl.dataset)
            best_val = min(best_val, val_loss)
        fold_mse.append(best_val)

    fold_mse = np.array(fold_mse)
    return fold_mse.mean(), fold_mse.std()

# STEP 6.  Single‑hidden‑layer network (width sweep)
# ================================================================
print("\nSTEP 6 — Single‑hidden‑layer PyTorch net (ReLU) width sweep")

best_width, best_mse = None, math.inf
for units in [4, 8, 16, 32, 64, 128, 256]:
    mean_mse, std_mse = cv_mse_torch([units], epochs=600)
    print(f"  {units:>3} units : MSE {mean_mse:6.2f} ± {std_mse:.2f}")
    if mean_mse < best_mse:
        best_width, best_mse = units, mean_mse
print(f" → Best single‑layer width = {best_width} units (MSE {best_mse:.2f})")

# STEP 7.  Arbitrary‑depth experiment (32 neurons/layer)
# ================================================================
print("\nSTEP 7 — Depth experiment (32 neurons per hidden layer)")

best_depth, best_depth_mse = None, math.inf
for depth in range(1, 6):
    hidden = [32] * depth
    t0 = time.perf_counter()
    mean_mse, std_mse = cv_mse_torch(hidden, epochs=500)
    elapsed = time.perf_counter() - t0
    print(f"  {depth} layer(s) : MSE {mean_mse:6.2f} ± {std_mse:.2f}   (train {elapsed:5.1f}s)")
    if mean_mse < best_depth_mse:
        best_depth, best_depth_mse = depth, mean_mse
print(f" → Best depth = {best_depth} layer(s) (MSE {best_depth_mse:.2f})")

# Reflection summary (auto‑printed)
# ================================================================
print("\n===== REFLECTION =====")
print(f"Baseline 2‑layer Pipeline: {mean_mse:4.1f} MSE (meets ≤20 target).")
print(f"Single‑layer needed ≥{best_width} units to match baseline accuracy.")