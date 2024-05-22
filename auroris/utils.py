import os
from io import BytesIO
from typing import ByteString
import fsspec

import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from PIL.Image import Image as ImageType

from sklearn.utils.multiclass import type_of_target

import datamol as dm


def is_regression(values: np.ndarray) -> bool:
    """Whether the input values are for regreesion"""
    target_type = type_of_target(values)
    if target_type == "continuous":
        return True
    elif target_type in ["binary", "multiclass"]:
        return False
    else:
        raise ValueError(f"Unsupported target type: {target_type}")


def fig2img(fig: Figure) -> ImageType:
    """Convert a Matplotlib figure to a PIL Image"""
    fig.canvas.draw()
    return Image.frombytes(
        "RGBA",
        fig.canvas.get_width_height(),
        fig.canvas.buffer_rgba(),
    )


def img2bytes(image: ImageType):
    """Convert png image to bytes"""
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()
    return image_bytes


def bytes2img(image_bytes: ByteString):
    """Convert bytes to PIL image"""
    image_stream = BytesIO(image_bytes)
    # Open the image using PIL
    image = Image.open(image_stream)
    return image


def _img_to_html_src(self, path: str):
    """
    Convert a path to a corresponding `src` attribute for an `<img />` tag.
    Currently only supports GCP and local paths.
    """
    protocol = dm.utils.fs.get_protocol(path)
    if protocol == "gs":
        return path.replace("gs://", "https://storage.googleapis.com/")
    elif protocol == "file":
        return os.path.relpath(path, self._destination)
    else:
        raise ValueError("We only support images hosted in GCP or locally")


def save_image(image: ImageType, path: str):
    with fsspec.open(path, "wb") as fd:
        image.save(fd, format="png")
