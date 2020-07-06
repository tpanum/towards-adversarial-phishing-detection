import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import vgg16

from .loss import _get_predictions


class WhiteNetModel(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        vgg16_model = vgg16(pretrained=True)

        self.features = vgg16_model.features
        for child in list(self.features.children()):
            for param in child.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Conv2d(512, embedding_size, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
        )

        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(-1, 512)

        return x


def _ensure_scalar(x):
    if isinstance(x, torch.Tensor) and len(x.shape) > 0:
        # when using multiple GPUs, nn.DataParallel yields a vector instead of a scalar
        return x.sum()

    return x


def loss_from_batch(
    model, batch, labels, loss_fn, embeddings=None, device=None
):
    timings = {}
    labels = labels.to(device)

    if embeddings is None:
        batch = batch.to(device)

        emb_time_start = time.time()
        embeddings = model.forward(batch)
        emb_time = time.time() - emb_time_start

        timings["emb_time"] = emb_time
    else:
        embeddings = embeddings.to(device)

    loss_time_start = time.time()
    loss = loss_fn(embeddings, labels)
    n_triplets = 0

    loss_time = time.time() - loss_time_start

    timings["loss_time"] = loss_time

    loss = _ensure_scalar(loss)
    n_triplets = _ensure_scalar(n_triplets)

    return loss, n_triplets, timings


def get_embeddings_from_loader(model, loader, device="cpu", attack=None):
    embeddings = []
    y = []

    for batch, labels in iter(loader):
        batch = batch.to(device)
        labels = labels.to(device)

        if attack is not None:
            batch = attack.forward(batch, labels, model)

        with torch.no_grad():
            emb = model.forward(batch)

        embeddings.append(emb)
        y.append(labels)

    embeddings = torch.cat(embeddings)
    y = torch.cat(y)

    return embeddings, y


def evaluate_acc(
    model,
    dataset,
    reference_dataset=None,
    batch_size=128,
    attack=None,
    device="cpu",
):
    pred_kwargs = {"device": device}
    model.eval()

    if not isinstance(dataset, DataLoader):
        dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    embs, y = get_embeddings_from_loader(model, dataset, device=device)

    if attack is not None:
        adv_embs, adv_y = get_embeddings_from_loader(
            model, dataset, device=device, attack=attack
        )

        # if not isinstance(reference_dataset, DataLoader):
        #     reference_dataset = DataLoader(
        #         reference_dataset, batch_size=batch_size
        #     )

        # compare_embs, compare_y = get_embeddings_from_loader(
        #     model, reference_dataset, device=device
        # )

        pred_y = _get_predictions(
            adv_embs, y, compare_embeddings=embs, **pred_kwargs
        )
        correct = (adv_y == pred_y).sum().item()

        pred_y_true = _get_predictions(embs, y, **pred_kwargs)
    else:
        pred_y = _get_predictions(embs, y, **pred_kwargs)
        correct = (pred_y == y).sum().item()

    return correct / len(y)


def train_batch(model, batch, labels, optimizer, loss_fn, device="cpu"):
    model.train()
    loss, n_samples, timings = loss_from_batch(
        model, batch, labels, loss_fn, device=device
    )

    backprop_time_start = time.time()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    backprop_time = time.time() - backprop_time_start

    timings["backprop_time"] = backprop_time

    return loss, n_samples, timings
