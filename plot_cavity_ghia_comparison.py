from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GHIA_U_Y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
GHIA_U_RE100 = np.array([1.0000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.00000])
GHIA_V_X = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
GHIA_V_RE100 = np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091, 0.09233, 0.00000])


def main():
    output_dir = Path("out")
    figure_dir = Path("figures")
    figure_dir.mkdir(exist_ok=True)

    data = np.load(output_dir / "re_00100.npz")
    x = data["x"]
    y = data["y"]
    u = data["u"]
    v = data["v"]

    x_mid = x.shape[1] // 2
    y_mid = y.shape[0] // 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(y[:, x_mid], u[:, x_mid], label="FEM solution", color="tab:red")
    axes[0].plot(GHIA_U_Y, GHIA_U_RE100, "x", color="black", label="Ghia et al.")
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("u(0.5, y)")
    axes[0].grid(linestyle="dashed", color="gray", alpha=0.6)
    axes[0].legend()

    axes[1].plot(x[y_mid, :], v[y_mid, :], label="FEM solution", color="tab:blue")
    axes[1].plot(GHIA_V_X, GHIA_V_RE100, "x", color="black", label="Ghia et al.")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("v(x, 0.5)")
    axes[1].grid(linestyle="dashed", color="gray", alpha=0.6)
    axes[1].legend()

    fig.suptitle("Ghia Comparison, Re = 100")
    fig.tight_layout()
    fig.savefig(figure_dir / "ghia_comparison_re100.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
