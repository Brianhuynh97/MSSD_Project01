from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np


ParsedValue: TypeAlias = int | float | str | list[str]


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


def _parse_value(raw: str) -> ParsedValue:
    text = raw.strip()
    if "," in text:
        return [item.strip() for item in text.split(",") if item.strip()]
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _require_value(values: dict[str, ParsedValue], key: str) -> ParsedValue:
    if key not in values:
        raise KeyError(f"Missing required parameter: {key}")
    return values[key]


def _require_int(values: dict[str, ParsedValue], key: str) -> int:
    value = _require_value(values, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected integer value for {key!r}, got {type(value).__name__}")
    return value


def _require_float(values: dict[str, ParsedValue], key: str) -> float:
    value = _require_value(values, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Expected float value for {key!r}, got {type(value).__name__}")
    return float(value)


def _require_str(values: dict[str, ParsedValue], key: str) -> str:
    value = _require_value(values, key)
    if not isinstance(value, str):
        raise TypeError(f"Expected string value for {key!r}, got {type(value).__name__}")
    return value


def _require_str_list(values: dict[str, ParsedValue], key: str) -> list[str]:
    value = _require_value(values, key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"Expected comma-separated list for {key!r}, got {type(value).__name__}")
    return value


def read_parameters(filename: str) -> Parameters:
    values: dict[str, ParsedValue] = {}
    for line in Path(filename).read_text(encoding="utf-8").splitlines():
        head = line.split("#", 1)[0].strip()
        if not head:
            continue
        key, raw_value = [part.strip() for part in head.split("=", 1)]
        values[key] = _parse_value(raw_value)

    reynolds_numbers = [int(value) for value in _require_str_list(values, "reynolds_numbers")]
    return Parameters(
        mesh_cells=_require_int(values, "mesh_cells"),
        sample_points=_require_int(values, "sample_points"),
        time_step=_require_float(values, "time_step"),
        final_time=_require_float(values, "final_time"),
        lid_velocity=_require_float(values, "lid_velocity"),
        reynolds_numbers=reynolds_numbers,
        newton_rtol=_require_float(values, "newton_rtol"),
        newton_atol=_require_float(values, "newton_atol"),
        newton_max_it=_require_int(values, "newton_max_it"),
        ksp_type=_require_str(values, "ksp_type"),
        pc_type=_require_str(values, "pc_type"),
        quiver_stride=_require_int(values, "quiver_stride"),
        frame_stride=_require_int(values, "frame_stride"),
        gif_fps=_require_int(values, "gif_fps"),
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
