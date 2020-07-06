import numpy as np

from hsluv import hsluv_to_rgb, rgb_to_hsluv
from PIL import Image, ImageChops, ImageOps


def img_to_matrix(img: Image) -> np.ndarray:
    return np.array(img)[:, :, :3]


class HSLPerturb:
    def __init__(self, h_step=0, s_step=0, l_step=0):
        self.mul = np.ones(3) + np.array([h_step, s_step, l_step])

    def perturb(self, img):
        raw = img_to_matrix(img)

        perturbed = raw / 255.0
        w, h, c = perturbed.shape

        for i in range(w):
            for j in range(h):
                px_color = perturbed[i, j]
                hsl = np.array(rgb_to_hsluv(px_color))

                hsl *= self.mul
                hsl[0] = np.clip(hsl[0], 0, 360)
                hsl[1:] = np.clip(hsl[1:], 0, 100)

                perturbed[i, j] = hsluv_to_rgb(hsl)

        perturbed *= 255.0

        raw_img = Image.fromarray(raw.astype("uint8"), "RGB")
        perturbed_img = Image.fromarray(perturbed.astype("uint8"), "RGB")
        noise_img = ImageChops.difference(raw_img, perturbed_img)
        noise_img = Image.fromarray(img_to_matrix(noise_img) * 50, "RGB")
        # noise_img = ImageOps.invert(noise_img.convert("L")).convert("RGB")
        # print((img_to_matrix(noise_img) * 10).max())
        # noise_img = Image.fromarray(, "RGB")

        return raw_img, perturbed_img, noise_img
