"""Legacy-compatible adaptive prediction mixer for BRIDGE.

The public module name is adaptive prediction mixer. The historical class and
function names are retained so completed scripts and checkpoints remain usable.
"""

import numpy as np
import torch
from torch import nn


class AdaptivePredictionMixer(nn.Module):
    """Legacy class name for the BRIDGE adaptive prediction mixer."""

    def __init__(self, n_tissues):
        super().__init__()
        self.tissue = nn.Embedding(n_tissues, 8)
        self.net = nn.Sequential(
            nn.Linear(8 + 1, 16),
            nn.GELU(),
            nn.Linear(16, 2),
        )

    def forward(self, expert_stack, target_tissue, known_counts):
        features = torch.cat([self.tissue(target_tissue), known_counts.float().unsqueeze(1) / 64.0], dim=1)
        weights = torch.softmax(self.net(features), dim=1)
        pred = (expert_stack * weights.unsqueeze(-1)).sum(dim=1)
        return pred, weights


def train_adaptive_mixer(train_experts, train_true, train_tissue, train_counts, val_experts, val_true, val_tissue, val_counts, n_tissues, epochs, batch_size, lr, device):
    gate = AdaptivePredictionMixer(n_tissues).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-4)
    x = torch.tensor(train_experts, dtype=torch.float32)
    y = torch.tensor(train_true, dtype=torch.float32)
    t = torch.tensor(train_tissue, dtype=torch.long)
    c = torch.tensor(train_counts, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(x, y, t, c)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    best_state = None
    best_val = None
    for _ in range(int(epochs)):
        gate.train()
        for xb, yb, tb, cb in loader:
            pred, _ = gate(xb.to(device), tb.to(device), cb.to(device))
            loss = torch.nn.functional.smooth_l1_loss(pred, yb.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_pred, _ = predict_with_adaptive_mixer(gate, val_experts, val_tissue, val_counts, batch_size, device)
        val_mae = float(np.abs(val_pred - val_true).mean())
        if best_val is None or val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
    gate.load_state_dict(best_state)
    return gate, best_val


@torch.no_grad()
def predict_with_adaptive_mixer(gate, experts, tissue, counts, batch_size, device):
    gate.eval()
    x = torch.tensor(experts, dtype=torch.float32)
    t = torch.tensor(tissue, dtype=torch.long)
    c = torch.tensor(counts, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(x, t, c)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    weights = []
    for xb, tb, cb in loader:
        pred, w = gate(xb.to(device), tb.to(device), cb.to(device))
        preds.append(pred.cpu().numpy())
        weights.append(w.cpu().numpy())
    return np.vstack(preds).astype(np.float32), np.vstack(weights).astype(np.float32)


AdaptivePredictionMixer = AdaptivePredictionMixer
train_adaptive_prediction_mixer = train_adaptive_mixer
predict_with_adaptive_prediction_mixer = predict_with_adaptive_mixer


