"""BRIDGE sample-matching encoder and contrastive objective."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class RecordDataset(Dataset):
    def __init__(self, records, expr, meta, tissue_means, max_known=64, return_target_index=False):
        self.records = records
        self.expr = expr
        self.tissue = meta["tissue_idx"].to_numpy()
        self.tissue_means = tissue_means
        self.max_known = max_known
        self.return_target_index = bool(return_target_index)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        known = list(r["known_indices"])[: self.max_known]
        known_expr = np.zeros((self.max_known, self.expr.shape[1]), dtype=np.float32)
        known_tissue = np.zeros(self.max_known, dtype=np.int64)
        known_mask = np.zeros(self.max_known, dtype=np.float32)
        for i, j in enumerate(known):
            t = int(self.tissue[int(j)])
            known_expr[i] = self.expr[int(j)] - self.tissue_means[t]
            known_tissue[i] = t
            known_mask[i] = 1.0
        target_idx = int(r["idx"])
        target_tissue = int(self.tissue[target_idx])
        target_expr = self.expr[target_idx]
        target_resid = target_expr - self.tissue_means[target_tissue]
        values = (
            torch.tensor(known_expr, dtype=torch.float32),
            torch.tensor(known_tissue, dtype=torch.long),
            torch.tensor(known_mask, dtype=torch.float32),
            torch.tensor(target_tissue, dtype=torch.long),
            torch.tensor(target_resid, dtype=torch.float32),
            torch.tensor(target_expr, dtype=torch.float32),
            torch.tensor(self.tissue_means[target_tissue], dtype=torch.float32),
        )
        if self.return_target_index:
            return values + (torch.tensor(target_idx, dtype=torch.long),)
        return values


class SampleMatchingEncoder(nn.Module):
    """Encode donor evidence and candidate profiles for sample matching."""

    def __init__(self, n_genes, n_tissues, emb_dim=128):
        super().__init__()
        self.tissue = nn.Embedding(n_tissues, 32)
        self.expr_encoder = nn.Sequential(
            nn.Linear(n_genes, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, emb_dim),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(emb_dim + 32, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.query_proj = nn.Sequential(
            nn.Linear(emb_dim + 32, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def encode_expr(self, residual):
        return F.normalize(self.expr_encoder(residual), dim=-1)

    def encode_query(self, known_expr, known_tissue, known_mask, target_tissue):
        b, m, g = known_expr.shape
        z_known = self.encode_expr(known_expr.reshape(b * m, g)).reshape(b, m, -1)
        evidence_vectors = self.context_proj(torch.cat([z_known, self.tissue(known_tissue)], dim=-1))
        mask = known_mask.unsqueeze(-1)
        donor_state = (evidence_vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        z = self.query_proj(torch.cat([donor_state, self.tissue(target_tissue)], dim=-1))
        return F.normalize(z, dim=-1)

    def forward(self, known_expr, known_tissue, known_mask, target_tissue, target_resid):
        return self.encode_query(known_expr, known_tissue, known_mask, target_tissue), self.encode_expr(target_resid)


def contrastive_loss(z_query, z_target, tau=0.07, target_indices=None):
    """Symmetric InfoNCE with records sharing a target treated as positives."""

    logits = z_query @ z_target.T / tau
    if target_indices is not None:
        target_indices = torch.as_tensor(target_indices, device=logits.device)
        if target_indices.ndim != 1 or len(target_indices) != len(z_query):
            raise ValueError("target_indices must contain one value per query-target pair")
        positive = target_indices[:, None].eq(target_indices[None, :])

        def multi_positive_loss(values, mask):
            numerator = torch.logsumexp(values.masked_fill(~mask, -torch.inf), dim=1)
            denominator = torch.logsumexp(values, dim=1)
            return -(numerator - denominator).mean()

        return (multi_positive_loss(logits, positive) + multi_positive_loss(logits.T, positive.T)) / 2

    labels = torch.arange(len(z_query), device=z_query.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


