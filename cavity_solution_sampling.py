from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def configure_cache_dirs(project_root: Path):
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplconfig"))


def sample_functions_on_grid(domain, velocity, pressure, sample_points: int):
    from dolfinx import geometry

    xs = np.linspace(0.0, 1.0, sample_points)
    ys = np.linspace(0.0, 1.0, sample_points)
    xy = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
    points = np.column_stack([xy, np.zeros(len(xy), dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    local_points = []
    local_cells = []
    local_indices = []

    for index, point in enumerate(points):
        candidates = geometry.compute_collisions_points(tree, point[None, :])
        colliding = geometry.compute_colliding_cells(domain, candidates, point[None, :])
        if len(colliding.links(0)) > 0:
            local_points.append(point)
            local_cells.append(colliding.links(0)[0])
            local_indices.append(index)

    velocity_values = None
    pressure_values = None
    if local_points:
        local_points_array = np.asarray(local_points, dtype=np.float64)
        local_cells_array = np.asarray(local_cells, dtype=np.int32)
        velocity_values = velocity.eval(local_points_array, local_cells_array)
        pressure_values = pressure.eval(local_points_array, local_cells_array)

    gathered_velocity = domain.comm.gather((local_indices, velocity_values), root=0)
    gathered_pressure = domain.comm.gather((local_indices, pressure_values), root=0)

    if domain.comm.rank != 0:
        return None

    u = np.zeros((sample_points, sample_points), dtype=np.float64)
    v = np.zeros((sample_points, sample_points), dtype=np.float64)
    p = np.zeros((sample_points, sample_points), dtype=np.float64)
    filled = np.zeros((sample_points, sample_points), dtype=bool)

    for (indices, values_u), (_, values_p) in zip(gathered_velocity, gathered_pressure):
        if values_u is None:
            continue
        for local_pos, global_index in enumerate(indices):
            j = global_index // sample_points
            i = global_index % sample_points
            if filled[j, i]:
                continue
            u[j, i] = values_u[local_pos, 0]
            v[j, i] = values_u[local_pos, 1]
            p[j, i] = values_p[local_pos, 0]
            filled[j, i] = True

    x_grid, y_grid = np.meshgrid(xs, ys)
    return x_grid, y_grid, u, v, p


def evaluate_function_at_points(function, xy_points: np.ndarray):
    from dolfinx import geometry

    domain = function.function_space.mesh
    points = np.column_stack([xy_points, np.zeros(len(xy_points), dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    local_points = []
    local_cells = []
    local_indices = []

    for index, point in enumerate(points):
        candidates = geometry.compute_collisions_points(tree, point[None, :])
        colliding = geometry.compute_colliding_cells(domain, candidates, point[None, :])
        if len(colliding.links(0)) > 0:
            local_points.append(point)
            local_cells.append(colliding.links(0)[0])
            local_indices.append(index)

    values = None
    if local_points:
        values = function.eval(np.asarray(local_points, dtype=np.float64), np.asarray(local_cells, dtype=np.int32))

    gathered = domain.comm.gather((local_indices, values), root=0)
    if domain.comm.rank != 0:
        return None

    result = np.full((len(xy_points), values.shape[1] if values is not None and values.ndim > 1 else 1), np.nan)
    for indices, part in gathered:
        if part is None:
            continue
        for local_pos, global_index in enumerate(indices):
            result[global_index, : part.shape[1] if part.ndim > 1 else 1] = (
                part[local_pos] if part.ndim > 1 else np.array([part[local_pos]])
            )
    return result


def sample_velocity_centerlines(velocity, npoints: int = 241):
    y_line = np.linspace(0.02, 0.98, npoints)
    x_line = np.linspace(0.02, 0.98, npoints)
    vertical_points = np.column_stack([np.full_like(y_line, 0.5), y_line])
    horizontal_points = np.column_stack([x_line, np.full_like(x_line, 0.5)])

    vertical_values = evaluate_function_at_points(velocity, vertical_points)
    horizontal_values = evaluate_function_at_points(velocity, horizontal_points)
    if vertical_values is None or horizontal_values is None:
        return None

    return {
        "vertical_y": y_line,
        "vertical_u": vertical_values[:, 0],
        "horizontal_x": x_line,
        "horizontal_v": horizontal_values[:, 1],
    }
