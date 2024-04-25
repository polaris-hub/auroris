import numpy as np
from sklearn.utils.multiclass import type_of_target


def is_regression(values: np.ndarray):
    target_type = type_of_target(values)
    if target_type == "continuous":
        return True
    elif target_type in ["binary", "multiclass"]:
        return False
    else:
        raise ValueError(f"Unsupported target type: {target_type}")
