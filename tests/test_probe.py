import torch
from torch import nn

from eeg_benchmark.probe import score_linear_probe


def test_score_linear_probe_returns_accuracy_for_head_predictions() -> None:
    head = nn.Linear(2, 2)
    head.weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    head.bias.data.zero_()

    embeddings = torch.tensor(
        [
            [2.0, 1.0],
            [1.0, 2.0],
            [3.0, 1.0],
            [1.0, 3.0],
        ]
    )
    labels = torch.tensor([0, 0, 0, 1])

    assert score_linear_probe(head, embeddings, labels, torch.device("cpu")) == 0.75
