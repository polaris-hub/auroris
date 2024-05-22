import os
from io import BytesIO
from typing import ByteString

import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from PIL.Image import Image as ImageType
import fsspec

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
    if isinstance(fig, Figure):
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


def path2url(path: str, destination: str):
    """
    Convert path to an local or remote url for html report.
    Currently, only GCP is supported.
    """
    if not os.path.isfile(path):
        if path.startswith("gs://"):
            return path.replace("gs://", "https://storage.googleapis.com/")
        else:
            raise ValueError("Only GCP path is supported.")
    else:
        return os.path.relpath(path, destination)


def save_image(image: ImageType, path: str, destination: str):
    """Save image to local and remote path"""
    if dm.fs.is_local_path(destination):
        image.save(path)
    else:
        # Lu: couldn't find a way to save image directly to remote path
        # convert to bytes
        image_bytes = img2bytes(image)
        # save bytes as image to remote path
        with fsspec.open(path, "wb") as f:
            f.write(image_bytes)
