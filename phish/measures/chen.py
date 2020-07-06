"""Normalized Compression Distance
   from "An Anti-Phishing System Employing Diffused Information" by Chen et al.
"""

import io
import lzma

from PIL import Image


class NormalizedCompressionDistance:
    def __init__(self, algo=lzma.compress):
        self.C = algo

    def get_distance(self, x: bytes, y: bytes, xy: bytes) -> float:
        C_x = len(self.C(x))
        C_y = len(self.C(y))
        C_xy = len(self.C(x + y))

        dist = (C_xy - min(C_x, C_y)) / max(C_x, C_y)

        return dist

    def img_to_bytes(self, img: Image):
        buffer = io.BytesIO()
        img.save(buffer, format="TIFF")
        data = buffer.getvalue()

        return data

    def measure(self, img1: Image, img2: Image) -> float:
        joint = Image.new(
            "RGB", (min(img1.width, img2.width), img1.height + img2.height)
        )
        joint.paste(img1, (0, 0))
        joint.paste(img2, (0, img1.height))

        x, y, xy = (
            self.img_to_bytes(img1),
            self.img_to_bytes(img2),
            self.img_to_bytes(joint),
        )

        return 1 - self.get_distance(x, y, xy)
