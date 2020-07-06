import argparse
import os.path as osp
import pathlib
import string
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from phish.measures.chen import NormalizedCompressionDistance
from phish.measures.linalg import Norm0, Norm2, NormInf
from phish.transform import HSLPerturb
from PIL import Image
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="assets/websites",
        help="default: assets/websites",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="perturb_summary.png",
        help="default: perturb_summary.png",
    )
    parser.add_argument(
        "--h-step", type=float, default="0", help="default: 0",
    )
    parser.add_argument(
        "--s-step", type=float, default="0", help="default: 0",
    )
    parser.add_argument(
        "--l-step", type=float, default="0", help="default: 0",
    )
    parser.add_argument(
        "--measures",
        type=str,
        default=["ncd", "l2"],
        choices=["ncd", "l0", "l2", "linf"],
        nargs="+",
        help="default: ncd,l2",
    )

    return parser.parse_args()


def main():
    args = get_args()

    if args.output_file.endswith(".pgf"):
        # adjust for pgf format, that is suitable for latex
        matplotlib.use("pgf")
        matplotlib.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
            }
        )

    attacks = {
        "HSL Perturbed": HSLPerturb(
            h_step=args.h_step, s_step=args.s_step, l_step=args.l_step
        )
    }

    images = {}
    if osp.isdir(args.input):
        for f in list(pathlib.Path(args.input).glob("*.png")):
            base = osp.basename(f)
            base, _ = osp.splitext(base)
            images[base] = [("Original", f)]
    elif osp.isfile(args.input):
        basename = args.input[:-4]
        images[basename] = [("Original", args.input)]
        pass
    else:
        raise ValueError("input is not file or directory")

    for name, imgs in images.items():
        for i, img in enumerate(imgs):
            img_name, path = img
            imgs[i] = (img_name, Image.open(path))

        for name, attack in attacks.items():
            img_org = imgs[0][1]
            img_raw, img_perturbed, img_noise = attack.perturb(img_org)

            imgs[0] = ("Original", img_raw)
            imgs.append(("Perturbation ($\\times$ $50$)", img_noise))
            imgs.append((name, img_perturbed))

    measures = {
        "ncd": NormalizedCompressionDistance(),  # Chen et al, 2014
        "l0": Norm0(),
        "l2": Norm2(),
        "linf": NormInf(),
    }

    gridspec = {
        "width_ratios": [1, 0.1, 1.0, 0.4, 1.0],
    }

    fig, axes = plt.subplots(len(images), 3 + 2, gridspec_kw=gridspec)
    diff_sims = defaultdict(list)

    for i, (name, imgs) in tqdm(enumerate(images.items()), unit="img"):
        org_measures = {}

        for j, txt in zip([1, 3], ["+", "$\\times$ $0.02$ ="]):
            ax = axes[i, j]
            ax.imshow(np.zeros((80, 80, 3), dtype=np.uint8) + 255)
            txt = ax.text(
                0.5,
                0.5,
                txt,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            txt.set_clip_on(False)
            ax.axis("off")

        for j, img in zip([0, 2, 4], imgs):
            imgkind, imgsrc = img
            ax = axes[i, j]

            if i == 0:
                ax.set_title(imgkind, fontsize="medium")

            if j == 0:
                ax.set_ylabel(
                    name,
                    rotation=0,
                    horizontalalignment="right",
                    verticalalignment="center",
                )

            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.imshow(imgsrc)

            if imgkind == "Noise":
                continue

            for mname in args.measures:
                f = measures[mname]

                org_imgsrc = imgs[0][1]
                val = f.measure(org_imgsrc, imgsrc)
                org_img_name = f"{mname}_base"

                if imgkind == "Original":
                    org_measures[org_img_name] = val
                    continue

                diff = val - org_measures[org_img_name]
                diff_sims[mname].append(diff)

    for mname, diffs in diff_sims.items():
        diffs = np.array(diffs)
        print(
            f"[{mname}] Differences of metric\n  mean: {diffs.mean():.2f}\n  std: {diffs.std():.2f}"
        )

    for ax in axes.reshape(-1):
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])

    fig.tight_layout(h_pad=-12, w_pad=0)

    fig.savefig(args.output_file, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
