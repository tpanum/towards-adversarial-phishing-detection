import argparse
import json
import os

import numpy as np

import torch
from phish import attacks
from phish.datasets import (
    DEFAULT_TRANSFORMS,
    ImageFolderLMDB,
    ImageTripletDataset,
)
from phish.utils import get_logger, load_checkpoint, save_checkpoint
from phish.whitenet import WhiteNetModel
from pytorch_metric_learning import losses, miners, samplers
from pytorch_metric_learning.utils import AccuracyCalculator
from torchvision.datasets import ImageFolder

logger = get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="default: 0")
    parser.add_argument(
        "--train-path",
        type=str,
        default="./data/train",
        help="default: ./data/train",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="./data/test",
        help="default: ./data/test",
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="default: 2e-5")
    parser.add_argument(
        "--margin", type=float, default=2.2, help="default: 2.2"
    )
    parser.add_argument(
        "--n_batches_random", type=int, default=21000, help="default: 21000"
    )
    parser.add_argument(
        "--n_triplets_per_batch_random",
        type=int,
        default=32,
        help="default: 32",
    )
    parser.add_argument(
        "--n_batches_hard", type=int, default=18000, help="default: 18000"
    )
    parser.add_argument(
        "--n_triplets_per_batch_hard",
        type=int,
        default=75,
        help="default: 75",
    )
    parser.add_argument(
        "--n_batches_adv", type=int, default=3000, help="default: 30000"
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="checkpoint",
        help="default: checkpoint",
    )
    parser.add_argument("--n_workers", type=int, default=4, help="default: 4")
    parser.add_argument(
        "--adv-train", action="store_true", help="default: false"
    )
    parser.add_argument("--cuda", action="store_true", help="default: false")

    # attacks
    parser.add_argument(
        "--epsilon",
        type=float,
        default=[0.005, 0.01],
        nargs="+",
        help="default: [0.005, 0.01]",
    )

    return parser.parse_args()


def get_image_dataset(path):
    last_dir = path.split(os.sep)[-1]
    lmdb_file = os.path.join(path, last_dir + ".lmdb")

    if os.path.isfile(lmdb_file):
        return ImageFolderLMDB(lmdb_file)
    else:
        return ImageFolder(path, transform=DEFAULT_TRANSFORMS)


def get_embeddings(model, dataset, batch_size=128, device=None):
    loader = torch.utils.data.DataLoader(dataset, batch_size=128)

    embeddings = []
    labels = []

    model_device = next(model.parameters()).device

    for (X, y) in iter(loader):
        X, y = X.to(model_device), y.to(model_device)
        embs = model(X)

        embeddings.append(embs)
        labels.append(y)

    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)

    if device is not None:
        return embeddings.to(device), labels.to(device)

    return embeddings, labels


def evaluate(eval_datasets, model, epsilons, loss_adv_fn, device="cpu"):
    acc_calc = AccuracyCalculator()
    metrics = {}

    for name, dataset in eval_datasets.items():
        with torch.no_grad():
            embs, labels = get_embeddings(model, dataset, device="cpu")
        embs, labels = embs.numpy(), labels.numpy()

        metrics[name] = acc_calc.get_accuracy(embs, embs, labels, labels, True)

        if name == "test":
            sampler = samplers.MPerClassSampler(labels, 2)

            for epsilon in epsilons:
                adv_embs, adv_labels = [], []
                fgsm = attacks.FastGradientSignMethod(epsilon)

                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=256, sampler=sampler
                )

                for (X, y) in iter(loader):
                    X, y = X.to(device), y.to(device)
                    X = fgsm(model, loss_adv_fn, X, y)

                    with torch.no_grad():
                        embeddings = model(X)
                        adv_embs.append(embeddings)

                    adv_labels.append(y)

                adv_embs = torch.cat(adv_embs).cpu().numpy()
                adv_labels = torch.cat(adv_labels).cpu().numpy()

                metrics[f"adv@{epsilon}_{name}"] = acc_calc.get_accuracy(
                    adv_embs, embs, adv_labels, labels, True
                )

    return metrics


def main():
    args = get_args()
    logger.info(f"Args: {args}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if torch.cuda.is_available() and args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = WhiteNetModel().to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    batch_num, epoch = load_checkpoint(
        model, opt, filename=args.checkpoint_file
    )

    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        opt, lambda batch: 0.99
    )

    loss_fn = losses.TripletMarginLoss(
        margin=args.margin,
        triplets_per_anchor=args.n_triplets_per_batch_random,
    )

    loss_adv_fn = losses.TripletMarginLoss(
        margin=args.margin, triplets_per_anchor="all"
    )

    train_dataset = get_image_dataset(args.train_path)
    test_dataset = get_image_dataset(args.test_path)

    eval_datasets = {
        "train": train_dataset,
        "test": test_dataset,
    }

    _, train_labels = get_embeddings(model, train_dataset)
    # initial training (random)
    sampler = samplers.MPerClassSampler(train_labels.cpu().numpy(), 2)

    logger.info("Epoch \t Batch \t Phase \t Loss")
    while batch_num < args.n_batches_random:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=args.n_workers,
            batch_size=args.n_triplets_per_batch_random,
            sampler=sampler,
            pin_memory=True,
        )

        train_losses = []

        for X, y in iter(train_loader):
            X, y = X.to(device), y.to(device)
            embs = model(X)

            loss = loss_fn(embs, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.detach().item())

            if batch_num > 0 and batch_num % 300 == 0:
                scheduler.step()

            batch_num += 1

            if batch_num % 200 == 0:
                mean_loss = np.array(train_losses).mean()
                logger.info(
                    f"{epoch} \t {batch_num} \t random \t {mean_loss:.4f}"
                )

            if batch_num >= args.n_batches_random:
                break

        epoch += 1

        save_checkpoint(
            model.state_dict(),
            opt.state_dict(),
            batch_num,
            epoch,
            filename=args.checkpoint_file,
        )

    miner = miners.TripletMarginMiner(
        margin=args.margin, type_of_triplets="hard"
    )
    loss_fn = losses.TripletMarginLoss(margin=args.margin)

    n_batches_hard = args.n_batches_random + args.n_batches_hard
    while batch_num < n_batches_hard:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=args.n_workers,
            batch_size=args.n_triplets_per_batch_hard,
            sampler=sampler,
            pin_memory=True,
        )

        train_losses = []

        for X, y in iter(train_loader):
            X, y = X.to(device), y.to(device)
            embs = model(X)

            hard_triplets = miner(embs, y)
            loss = loss_fn(embs, y, hard_triplets)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.detach().item())

            if batch_num > 0 and batch_num % 300 == 0:
                scheduler.step()

            batch_num += 1

            if batch_num % 200 == 0:
                mean_loss = np.array(train_losses).mean()
                logger.info(
                    f"{epoch} \t {batch_num} \t hard \t {mean_loss:.4f}"
                )

            if batch_num >= n_batches_hard:
                break

        epoch += 1

        save_checkpoint(
            model.state_dict(),
            opt.state_dict(),
            batch_num,
            epoch,
            filename=args.checkpoint_file,
        )

    save_checkpoint(
        model.state_dict(),
        opt.state_dict(),
        batch_num,
        epoch,
        filename=f"final-{args.checkpoint_file}",
    )

    org_metrics = evaluate(
        eval_datasets, model, args.epsilon, loss_adv_fn, device=device
    )
    for name, m in org_metrics.items():
        prec = m["precision_at_1"]
        logger.info(f"\t org-{name} \t precesion_at_1: {prec}")

    attack = attacks.FastGradientSignMethod(alpha_min=0.003, alpha_max=0.01)
    n_batches_adv = n_batches_hard + args.n_batches_adv
    n_adv_per_batch = int(args.n_triplets_per_batch_hard / 2)
    while batch_num < n_batches_adv:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=args.n_workers,
            batch_size=args.n_triplets_per_batch_hard,
            sampler=sampler,
            pin_memory=True,
        )

        train_losses = []

        for X, y in iter(train_loader):
            X, y = X.to(device), y.to(device)

            # select half of training points
            batch_size = X.size(0)
            n_adv_examples = int(batch_size / 2) + 1
            adv_idx = torch.randperm(batch_size)[:n_adv_examples].to(device)

            X_adv, y_adv = (
                X[adv_idx].detach(),
                y[adv_idx].detach(),
            )

            X[adv_idx] = attack(model, loss_adv_fn, X_adv, y_adv)

            embs = model(X)

            hard_triplets = miner(embs, y)
            loss = loss_fn(embs, y, hard_triplets)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.detach().item())

            if batch_num > 0 and batch_num % 300 == 0:
                scheduler.step()

            batch_num += 1

            if batch_num % 200 == 0:
                mean_loss = np.array(train_losses).mean()
                logger.info(
                    f"{epoch} \t {batch_num} \t adv \t {mean_loss:.4f}"
                )

            if batch_num >= n_batches_adv:
                break

        epoch += 1

        save_checkpoint(
            model.state_dict(),
            opt.state_dict(),
            batch_num,
            epoch,
            filename=args.checkpoint_file,
        )

    save_checkpoint(
        model.state_dict(),
        opt.state_dict(),
        batch_num,
        epoch,
        filename=f"final-adv-{args.checkpoint_file}",
    )

    robust_metrics = evaluate(
        eval_datasets, model, args.epsilon, loss_adv_fn, device=device
    )

    for name, m in robust_metrics.items():
        prec = m["precision_at_1"]
        logger.info(f"\t adv-{name} \t precesion_at_1: {prec}")

    metrics = {"org": org_metrics, "robust": robust_metrics}
    with open(f"{args.checkpoint_file}_results.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
