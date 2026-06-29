"""Linear-probe training and scoring helpers."""

import torch
from sklearn.metrics import accuracy_score
from torch import Tensor, nn


def train_linear_probe(
    train_embeddings: Tensor,
    train_labels: Tensor,
    *,
    n_classes: int,
    device: torch.device,
    epochs: int,
    lr: float,
) -> nn.Linear:
    """Train a linear classifier on frozen embeddings."""
    head = nn.Linear(train_embeddings.shape[1], n_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_embeddings = train_embeddings.to(device)
    train_labels = train_labels.to(device)

    head.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(head(train_embeddings), train_labels)
        loss.backward()
        optimizer.step()

    return head


@torch.inference_mode()
def score_linear_probe(
    head: nn.Linear,
    embeddings: Tensor,
    labels: Tensor,
    device: torch.device,
) -> float:
    """Score a trained linear probe with accuracy."""
    head.eval()
    predictions = head(embeddings.to(device)).argmax(dim=1).cpu()
    return float(accuracy_score(labels.cpu().numpy(), predictions.numpy()))


def evaluate_linear_probe(
    train_embeddings: Tensor,
    train_labels: Tensor,
    test_embeddings: Tensor,
    test_labels: Tensor,
    *,
    n_classes: int,
    device: torch.device,
    epochs: int,
    lr: float,
) -> tuple[float, float]:
    """Train and evaluate a linear probe on train and test embeddings."""
    head = train_linear_probe(
        train_embeddings,
        train_labels,
        n_classes=n_classes,
        device=device,
        epochs=epochs,
        lr=lr,
    )
    train_accuracy = score_linear_probe(head, train_embeddings, train_labels, device)
    test_accuracy = score_linear_probe(head, test_embeddings, test_labels, device)
    return train_accuracy, test_accuracy
