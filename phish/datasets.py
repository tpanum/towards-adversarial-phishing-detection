import os
import pickle
import time

import numpy as np

import lmdb
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

DEFAULT_TRANSFORMS = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)


# from: https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py
class ImageFolderLMDB(Dataset):
    def __init__(self, db_path):
        self.db_path = db_path
        self.env = lmdb.open(
            db_path,
            subdir=os.path.isdir(db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
            self.keys = pickle.loads(txn.get(b"__keys__"))
            self.class_to_idx = pickle.loads(txn.get(b"class_to_idx"))
            self.samples = pickle.loads(txn.get(b"samples"))

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pickle.loads(byteflow)
        tensor, label = unpacked[0], unpacked[1]

        return tensor, label

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.db_path + ")"


class ImageTripletDataset(Dataset):
    def __init__(self, image_dataset):
        self.image_dataset = image_dataset
        self.samples = image_dataset.samples
        self.class_to_idx = image_dataset.class_to_idx
        self.labels_set = set(image_dataset.class_to_idx.values())
        self.dist_fn = nn.PairwiseDistance(p=2)

        self.labels = torch.Tensor([r[1] for r in image_dataset.samples])
        self.label_to_indices = {
            label: torch.where(self.labels == label)[0]
            for label in image_dataset.class_to_idx.values()
        }

        assert sum(
            [len(idx) for idx in self.label_to_indices.values()]
        ) == len(image_dataset)

        self.init_random_mining()

    def init_random_mining(self):
        triplet_idx = []

        for c, idx in self.label_to_indices.items():
            n_idx = len(idx)

            if n_idx <= 1:
                continue

            inv_eye_vec = ~torch.eye(n_idx).bool().flatten()
            possible_pos_idx = idx.repeat(n_idx)[inv_eye_vec].view(
                -1, n_idx - 1
            )
            positives_idx = idx[
                torch.randint(len(possible_pos_idx), (n_idx,))
            ].long()

            possible_neg_idx = self.labels[self.labels != c]
            negatives_idx = possible_neg_idx[
                torch.randint(len(possible_neg_idx), (n_idx,))
            ].long()

            triplet_idx.append(
                torch.stack([idx, positives_idx, negatives_idx], dim=1)
            )

        self.triplet_idx = torch.cat(triplet_idx)
        self.triplet_count = len(self.triplet_idx)

    def init_hard_mining(self, model, device="cpu", batch_size=256):
        model.eval()
        train_loader = DataLoader(self.image_dataset, batch_size=batch_size)
        embeddings = []

        with torch.no_grad():
            for batch, _ in iter(train_loader):
                batch = batch.to(device)
                embs = model.forward(batch)

                embeddings.append(embs)

        embeddings = torch.cat(embeddings)
        n = len(embeddings)

        triplet_idx = []

        for c, idx in self.label_to_indices.items():
            n_idx = len(idx)

            if n_idx <= 1:
                continue

            pos_embs = embeddings[idx]
            pos_idx_arg = get_arg_by_dist(
                pos_embs, pos_embs, arg="max", device=device
            )
            positives_idx = idx[pos_idx_arg].long()

            neg = self.labels != c
            neg_embs = embeddings[neg]
            neg_idx_arg = get_arg_by_dist(
                pos_embs, neg_embs, arg="min", device=device
            )
            negatives_idx = torch.arange(n)[neg][neg_idx_arg].long()

            triplet_idx.append(
                torch.stack([idx, positives_idx, negatives_idx], dim=1)
            )

        self.triplet_idx = torch.cat(triplet_idx)
        self.triplet_count = len(self.triplet_idx)

    def sample_hard_triplet(self, index, label_anchor):
        class_idxs = self.label_to_indices[label_anchor]
        pos_idxs = class_idxs[class_idxs != index]  # remove anchor

        pos_dists = self.dist_fn(
            self.embeddings[pos_idxs], self.embeddings[index]
        )
        positive_index = pos_idxs[torch.argmax(pos_dists)]

        neg_idxs = torch.ones(len(self.embeddings)) == 1
        neg_idxs[class_idxs] = False
        neg_idxs = torch.arange(len(self.embeddings))[neg_idxs]

        neg_dists = self.dist_fn(
            self.embeddings[neg_idxs], self.embeddings[index]
        )
        negative_index = neg_idxs[torch.argmin(neg_dists)]

        return positive_index, negative_index

    def __getitem__(self, idx):
        triplet = torch.stack(
            [self.image_dataset[i][0] for i in self.triplet_idx[idx]]
        )

        return triplet, []

    def __len__(self):
        return self.triplet_count


class ArgDistanceMeasure(nn.Module):
    def __init__(self, distance=nn.PairwiseDistance(p=2)):
        super().__init__()
        self.distance = distance

    def forward(self, a, b, n=1, batch_size=128, arg="min", device="cpu"):
        descending, default_val = {
            "min": (False, float("inf")),
            "max": (True, float("-inf")),
        }[arg]

        _, emb_size = a.shape
        dist = nn.PairwiseDistance(p=2)

        if isinstance(batch_size, tuple):
            a_batch_size, b_batch_size = batch_size
        else:
            a_batch_size, b_batch_size = (batch_size, batch_size)

        a_loader = DataLoader(a, batch_size=a_batch_size)
        b_loader = DataLoader(b, batch_size=b_batch_size)

        idx = []
        for a_batch in iter(a_loader):
            n_a = a_batch.shape[0]

            best_dists = torch.empty(n_a, n).fill_(default_val).to(device)
            best_idx = torch.zeros(n_a, n).long().to(device)

            for i, b_batch in enumerate(iter(b_loader)):
                n_b = b_batch.shape[0]

                batch_dists = dist(
                    a_batch.repeat(1, n_b).view(-1, emb_size),
                    b_batch.repeat(n_a, 1),
                ).view(-1, n_b)

                batch_org_idx = (torch.arange(n_b).repeat(n_a, 1) + i).to(
                    device
                )

                joint_dists = torch.cat([best_dists, batch_dists], dim=1)

                best_args = torch.argsort(
                    joint_dists, dim=1, descending=descending
                )[:, range(n)]

                joint_org_idx = torch.cat([best_idx, batch_org_idx], dim=1)

                best_dists = joint_dists.gather(1, best_args)
                best_idx = joint_org_idx.gather(1, best_args)

            idx.append(best_idx)

        return torch.cat(idx)
