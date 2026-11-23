import numpy as np
import torch
from torch.utils.data import Dataset

from bridge import training_utils
from bridge.sample_matching_expert import contrastive_loss


def test_retrieval_masks_every_memory_record_with_the_same_target():
    similarities = np.array(
        [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=np.float32
    )
    masked = training_utils.mask_same_target_similarities(
        similarities,
        query_target_indices=np.array([10, 20]),
        memory_target_indices=np.array([10, 10, 30]),
    )

    assert np.isneginf(masked[0, :2]).all()
    assert masked[0, 2] == similarities[0, 2]
    assert np.array_equal(masked[1], similarities[1])


def test_contrastive_loss_treats_duplicate_targets_as_multiple_positives():
    z_query = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    z_target = z_query.clone()
    target_indices = torch.tensor([7, 7, 9])

    multi_positive = contrastive_loss(
        z_query, z_target, tau=0.1, target_indices=target_indices
    )
    diagonal_only = contrastive_loss(z_query, z_target, tau=0.1)

    assert multi_positive < diagonal_only
    assert torch.isfinite(multi_positive)


class _OneQueryDataset(Dataset):
    records = [{"idx": 10}]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return (
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([0]),
            torch.tensor([1.0]),
            torch.tensor(0),
            torch.tensor([0.0, 0.0]),
            torch.tensor([9.0, 9.0]),
            torch.tensor([0.0, 0.0]),
        )


class _IdentityQueryModel:
    def eval(self):
        return self

    def encode_query(self, known_expr, known_tissue, known_mask, target_tissue):
        return known_expr[:, 0, :]


def test_predict_retrieval_cannot_select_a_duplicate_target_record():
    train_z = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32
    )
    train_expr = np.array(
        [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype=np.float32
    )

    _, predicted = training_utils.predict_retrieval(
        _IdentityQueryModel(),
        train_z,
        train_expr,
        _OneQueryDataset(),
        batch=1,
        k=1,
        device=torch.device("cpu"),
        exclude_self=True,
        train_target_indices=np.array([10, 10, 30]),
    )

    assert np.array_equal(
        predicted, np.array([[30.0, 30.0]], dtype=np.float32)
    )

