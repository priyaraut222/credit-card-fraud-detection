"""
Shared definitions used by BOTH train_models.py and app.py.

This file exists so the Autoencoder architecture and the exact feature
order used at training time can never silently drift from what's used
at inference time in the dashboard. Only one source of truth.
"""
import torch
import torch.nn as nn

# Column names in the original Kaggle "Credit Card Fraud Detection" CSV
PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]  # V1 ... V28

# Exact order of columns fed into every model. This matches what the
# original notebook produces: df.drop(['Time', 'Amount'], axis=1) after
# adding Amount_scaled and Time_scaled leaves columns in this order.
MODEL_FEATURE_ORDER = PCA_COLUMNS + ["Amount_scaled", "Time_scaled"]

MODELS_DIR = "models"


class Autoencoder(nn.Module):
    """Matches the architecture trained in the notebook: 30->20->14->7->14->20->30."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 14),
            nn.ReLU(),
            nn.Linear(14, 7),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(7, 14),
            nn.ReLU(),
            nn.Linear(14, 20),
            nn.ReLU(),
            nn.Linear(20, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_autoencoder(path: str, input_dim: int) -> "Autoencoder":
    model = Autoencoder(input_dim)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
