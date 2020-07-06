import numpy as np

from phish.transform import img_to_matrix
from PIL import Image


def dist_matrix(img1: Image, img2: Image) -> np.ndarray:
    m1 = img_to_matrix(img1)
    m2 = img_to_matrix(img2)

    return (m1 - m2).flatten()


class NormInf:
    def __init__(self):
        pass

    def measure(self, img1: Image, img2: Image) -> float:
        return np.linalg.norm(dist_matrix(img1, img2), ord=float("inf"))


class Norm2:
    def __init__(self):
        pass

    def measure(self, img1: Image, img2: Image) -> float:
        return np.linalg.norm(dist_matrix(img1, img2), ord=2)


class Norm0:
    def __init__(self):
        pass

    def measure(self, img1: Image, img2: Image) -> float:
        return np.linalg.norm(dist_matrix(img1, img2), ord=0)
