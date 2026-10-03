"""
PyTorch Autoencoder: architecture + loading.

Deliberately kept SEPARATE from model_utils.py. model_utils.py is imported
at the top of app.py on every single page render, so anything imported
there gets loaded into memory on every click. torch is a large, slow-to-
import library -- keeping it isolated here means app.py can import this
module lazily (inside a function, only when the Autoencoder/Anomaly
Detection page is actually used), which matters a lot on resource-limited
hosting like Streamlit Community Cloud's free tier.

train_models.py imports this normally at the top of the file, since it
needs torch for the entire training run anyway -- there's no benefit to
laziness there.
"""
import torch
import torch.nn as nn


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