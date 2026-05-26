from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Parameters:
    mesh_cells: int
    sample_points: int
    time_step: float
    final_time: float
    lid_velocity: float
    reynolds_numbers: list[int]
    newton_rtol: float
    newton_atol: float
    newton_max_it: int
    ksp_type: str
    pc_type: str
    quiver_stride: int
    frame_stride: int
    gif_fps: int


def _parse_value(raw: str):
    text = raw.strip()
    if "," in text:
        return [item.strip() for item in text.split(",") if item.strip()]
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def read_parameters(filename: str) -> Parameters:
    values: dict[str, object] = {}
    for line in Path(filename).read_text(encoding="utf-8").splitlines():
        head = line.split("#", 1)[0].strip()
        if not head:
            continue
        key, raw_value = [part.strip() for part in head.split("=", 1)]
        values[key] = _parse_value(raw_value)

    reynolds_numbers = [int(value) for value in values["reynolds_numbers"]]
    return Parameters(
        mesh_cells=int(values["mesh_cells"]),
        sample_points=int(values["sample_points"]),
        time_step=float(values["time_step"]),
        final_time=float(values["final_time"]),
        lid_velocity=float(values["lid_velocity"]),
        reynolds_numbers=reynolds_numbers,
        newton_rtol=float(values["newton_rtol"]),
        newton_atol=float(values["newton_atol"]),
        newton_max_it=int(values["newton_max_it"]),
        ksp_type=str(values["ksp_type"]),
        pc_type=str(values["pc_type"]),
        quiver_stride=int(values["quiver_stride"]),
        frame_stride=int(values["frame_stride"]),
        gif_fps=int(values["gif_fps"]),
    )


def write_case_data(output_dir: Path, reynolds_number: int, x, y, u, v, p, metadata: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"re_{reynolds_number:05d}.npz",
        x=x,
        y=y,
        u=u,
        v=v,
        p=p,
        speed=np.sqrt(u**2 + v**2),
    )
    with (output_dir / f"re_{reynolds_number:05d}.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)


def write_case_frame(frame_dir: Path, frame_index: int, time_value: float, x, y, u, v, p):
    frame_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        frame_dir / f"frame_{frame_index:04d}.npz",
        x=x,
        y=y,
        u=u,
        v=v,
        p=p,
        speed=np.sqrt(u**2 + v**2),
        time=time_value,
    )
