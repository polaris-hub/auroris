import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from PIL.Image import Image as ImageType
from sklearn.utils.multiclass import type_of_target
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from io import BytesIO


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
        "RGB",
        fig.canvas.get_width_height(),
        fig.canvas.tostring_rgb(),
    )


def fig2bytes(fig):
    """Convert image to bytes"""
    buffer = BytesIO()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buffer)

    # Get the bytes data
    image_data = buffer.getvalue()

    return image_data
