import logging
import os.path as osp
import sys

import numpy as np

import torch


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    np.random.seed(seed)


def get_logger(
    name,
    level=logging.INFO,
    log_format="[%(asctime)s][%(module)s.%(funcName)s] %(message)s",
    print_to_std=True,
):

    logging.getLogger().handlers = []

    if level is None:
        level = logging.INFO
    elif isinstance(level, str):
        level = logging.getLevelName(level)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if print_to_std:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(handler)

    return logger


def save_checkpoint(
    model_state, opt_state, batch, epoch, filename="checkpoint.pt"
):
    if filename.endswith(".pt") is False:
        filename = f"{filename}.pt"

    torch.save(
        {
            "batch": batch,
            "epoch": epoch,
            "model_state_dict": model_state,
            "opt_state_dict": opt_state,
        },
        filename,
    )


def load_checkpoint(model, opt, filename="checkpoint.pt"):
    if filename.endswith(".pt") is False:
        filename = f"{filename}.pt"

    if osp.isfile(filename) is False:
        return 0, 1

    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint["model_state_dict"])
    opt.load_state_dict(checkpoint["opt_state_dict"])

    return checkpoint["batch"], checkpoint["epoch"]
