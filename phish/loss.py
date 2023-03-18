import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class TripletLoss(nn.Module):
    def __init__(
        self, alpha=2.2, device="cpu"
    ):  # alpha is taken from the WhiteNet paper
        super(TripletLoss, self).__init__()
        self.alpha = alpha
        self.device = device

    def forward(self, anchors, positive, negative):
        anchors = anchors.to(self.device)
        positive = positive.to(self.device)
        negative = negative.to(self.device)

        sqrd_ap = (anchors - positive).pow(2).sum(dim=1)
        sqrd_an = (anchors - negative).pow(2).sum(dim=1)
        loss = sqrd_ap - sqrd_an + self.alpha
        loss = torch.max(loss, torch.zeros(loss.shape).to(self.device)).sum()

        return loss


# from: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/triplet_loss.py
class TripletLossEmbeddings(nn.Module):
    def __init__(self, alpha=2.2, squared=False, hard=False):
        super(TripletLossEmbeddings, self).__init__()
        self.alpha = alpha
        self.squared = squared

        self._forward = self.full_loss
        if hard:
            self._forward = self.hard_loss

    def full_loss(self, embeddings, labels, device="cpu"):
        pairwise_dist = _pairwise_distances(
            embeddings, embeddings, squared=self.squared
        )

        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_losses = (anchor_positive_dist - anchor_negative_dist + self.alpha).to(
            device
        )

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = _get_triplet_mask(labels, device=device)
        triplet_losses = mask.float() * triplet_losses

        # Remove negative losses (i.e. the easy triplets)
        triplet_losses[triplet_losses < 0] = 0

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_losses[triplet_losses > 1e-16]
        num_positive_triplets = valid_triplets.size(0)
        num_valid_triplets = mask.sum()

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = triplet_losses.sum() / (num_positive_triplets + 1e-16)

        return triplet_loss, num_valid_triplets

    def hard_loss(self, embeddings, labels, device="cpu"):
        pairwise_dist = _pairwise_distances(
            embeddings, embeddings, squared=self.squared
        )

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = _get_anchor_positive_triplet_mask(
            labels, device=device
        ).float()

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
            1.0 - mask_anchor_negative
        )

        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        tl = hardest_positive_dist - hardest_negative_dist + self.alpha
        tl[tl < 0] = 0
        triplet_loss = tl.mean()

        return triplet_loss

    def forward(self, *args, **kwargs):
        return self._forward(*args, **kwargs)


def _pairwise_distances(embeddings_a, embeddings_b, squared=False, batch_size=64):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """

    loader = DataLoader(embeddings_a, batch_size=batch_size)

    b_squared = torch.pow(embeddings_b, 2).sum(1)

    tmp = []
    for batch_a in iter(loader):
        dot_product = torch.matmul(batch_a, embeddings_b.t()).half()
        a_squared = torch.pow(batch_a, 2).sum(1)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = a_squared.unsqueeze(1) - 2.0 * dot_product + b_squared.unsqueeze(0)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances[distances < 0] = 0

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = distances.eq(0).float()
            distances = distances + mask * 1e-16

            distances = (1.0 - mask) * torch.sqrt(distances)

        tmp.append(distances)

    return torch.cat(tmp)


def _get_triplet_mask(labels, device="cpu"):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0)).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels.to(device) & distinct_indices.to(device)


def _get_anchor_positive_triplet_mask(labels, device="cpu"):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size(0)).bool().to(device)
    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)

    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))


def _get_predictions(embeddings, labels, compare_embeddings=None, device="cpu"):
    if compare_embeddings is not None:
        pairwise_dist = _pairwise_distances(compare_embeddings, embeddings)
    else:
        pairwise_dist = _pairwise_distances(embeddings, embeddings)
        indices_self = torch.eye(labels.size(0)).bool().to(device)
        pairwise_dist[indices_self] = float("inf")

    closest_idx = torch.argmin(pairwise_dist, dim=1)

    return labels[closest_idx]
