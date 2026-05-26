from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cavity_case_io import read_parameters


class _CanvasWithBufferRGBA(Protocol):
    def draw(self) -> None: ...

    def buffer_rgba(self) -> memoryview: ...


def load_case_data(output_dir: Path, reynolds_number: int):
    data = np.load(output_dir / f"re_{reynolds_number:05d}.npz")
    return data["x"], data["y"], data["u"], data["v"], data["p"], data["speed"]


def load_frame_data(frame_file: Path):
    data = np.load(frame_file)
    return data["x"], data["y"], data["u"], data["v"], data["p"], data["speed"], float(data["time"])


def render_case_figure(x, y, u, v, p, speed, reynolds_number: int, quiver_stride: int, time_value: float | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    speed_ax, pressure_ax = axes

    speed_plot = np.clip(speed, 0.0, 1.0)
    speed_contour = speed_ax.contourf(x, y, speed_plot, levels=np.linspace(0.0, 1.0, 21), cmap="viridis")
    stride = quiver_stride
    speed_ax.quiver(
        x[::stride, ::stride],
        y[::stride, ::stride],
        u[::stride, ::stride],
        v[::stride, ::stride],
        color="white",
        pivot="mid",
        angles="xy",
        scale_units="xy",
        scale=8,
        width=0.006,
    )
    speed_ax.set_title(f"Re = {reynolds_number}")
    speed_ax.set_aspect("equal")
    fig.colorbar(speed_contour, ax=speed_ax, fraction=0.046, pad=0.02)

    pressure_plot = p - float(np.mean(p))
    pressure_contour = pressure_ax.contourf(
        x,
        y,
        np.clip(pressure_plot, -0.1, 0.1),
        levels=np.linspace(-0.1, 0.1, 21),
        cmap="viridis",
    )
    pressure_ax.set_title(f"Pressure, Re = {reynolds_number}")
    pressure_ax.set_aspect("equal")
    fig.colorbar(pressure_contour, ax=pressure_ax, fraction=0.046, pad=0.02)

    title = f"Lid-Driven Cavity, Re = {reynolds_number}"
    if time_value is not None:
        title += f", t = {time_value:.2f}"
    fig.suptitle(title, fontsize=18)
    fig.tight_layout()
    return fig


def main():
    parameters = read_parameters("cavity_solver_config.txt")
    output_dir = Path("out")
    figure_dir = Path("figures")
    figure_dir.mkdir(exist_ok=True)

    for reynolds_number in parameters.reynolds_numbers:
        x, y, u, v, p, speed = load_case_data(output_dir, reynolds_number)
        fig = render_case_figure(x, y, u, v, p, speed, reynolds_number, parameters.quiver_stride)
        fig.savefig(figure_dir / f"lid_driven_cavity_re_{reynolds_number:05d}.png", dpi=150)
        plt.close(fig)

        frame_dir = output_dir / f"re_{reynolds_number:05d}_frames"
        frame_files = sorted(frame_dir.glob("frame_*.npz"))
        images = []
        for frame_file in frame_files:
            fx, fy, fu, fv, fp, fspeed, time_value = load_frame_data(frame_file)
            frame_fig = render_case_figure(
                fx, fy, fu, fv, fp, fspeed, reynolds_number, parameters.quiver_stride, time_value=time_value
            )
            canvas = cast(_CanvasWithBufferRGBA, frame_fig.canvas)
            canvas.draw()
            image = Image.fromarray(np.asarray(canvas.buffer_rgba())[:, :, :3])
            images.append(image)
            plt.close(frame_fig)

        if images:
            images[0].save(
                figure_dir / f"lid_driven_cavity_re_{reynolds_number:05d}.gif",
                save_all=True,
                append_images=images[1:],
                duration=int(1000 / max(parameters.gif_fps, 1)),
                loop=0,
            )


if __name__ == "__main__":
    main()
