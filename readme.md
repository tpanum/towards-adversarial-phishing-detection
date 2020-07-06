# Toward Adversarial Phishing Detection
This repository contains the implementation for the publication "Towards Adversarial Phishing Detection" that is presented at [USENIX CSET '20](https://www.usenix.org/conference/cset20/) and is designed to be used with the [conda package manager](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

## Prepare the environment
Prior to running the code, ensure you have a conda environment with the required dependencies.
```shell
conda env create -n <myenv> -f environment.yaml
conda activate <myenv>
```

## Models and weights
Weights for the [WhiteNet](https://arxiv.org/abs/1909.00300) is located in `assets/model-weights/whitenet.pt` (no adv. training) and `assets/model-weights/whitenet-adv.pt` (with adv. training).

These weighs were trained on a Tesla V100 GPU, using the `train.py` script and a data set of 37K websites across 2.5K domains.

## HSL Perturbation

If you wish to experiment with HSL Perturbation, run `plot.py` with your desired specification.
