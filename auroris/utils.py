from io import BytesIO

import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from PIL.Image import Image as ImageType
from sklearn.utils.multiclass import type_of_target


def is_regression(values: np.ndarray):
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
