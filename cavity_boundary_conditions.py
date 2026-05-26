import numpy as np


def top_boundary(x):
    return np.isclose(x[1], 1.0)


def wall_boundary(x):
    on_left = np.isclose(x[0], 0.0)
    on_right = np.isclose(x[0], 1.0)
    on_bottom = np.isclose(x[1], 0.0)
    return np.logical_or(np.logical_or(on_left, on_right), on_bottom)
