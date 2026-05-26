from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cavity_case_io import read_parameters
from cavity_solution_sampling import sample_velocity_centerlines
from solve_lid_driven_cavity import make_parameters_with_mesh, solve_case


MESH_LEVELS = [16, 24, 32, 40]
REFERENCE_MESH_CELLS = 64
TARGET_REYNOLDS = 100
RESULTS_DIR = Path("results/convergence")
FIGURES_DIR = Path("figures/convergence")
CONVERGENCE_TIME_STEP = 0.02
CENTERLINE_POINTS = 241


def line_l2_difference(values: np.ndarray, reference: np.ndarray) -> float:
    diff = values - reference
    return float(np.sqrt(np.nanmean(diff**2)))


def center_velocity_from_lines(lines: dict[str, np.ndarray]) -> tuple[float, float]:
    mid_vertical = len(lines["vertical_u"]) // 2
    mid_horizontal = len(lines["horizontal_v"]) // 2
    return float(lines["vertical_u"][mid_vertical]), float(lines["horizontal_v"][mid_horizontal])


def write_csv(rows: list[dict]):
    csv_path = RESULTS_DIR / "convergence_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "mesh_cells",
                "mesh_size_h",
                "ndofs",
                "reynolds_number",
                "mean_newton_iterations",
                "center_u",
                "center_v",
                "reference_center_u",
                "reference_center_v",
                "vertical_centerline_diff_to_reference",
                "horizontal_centerline_diff_to_reference",
                "vertical_observed_order",
                "horizontal_observed_order",
                "centerline_points",
                "time_step",
                "final_time",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def write_plot(rows: list[dict]):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURES_DIR / "convergence_summary.png"
    mesh_sizes = [row["mesh_size_h"] for row in rows]
    vertical_errors = [row["vertical_centerline_diff_to_reference"] for row in rows]
    horizontal_errors = [row["horizontal_centerline_diff_to_reference"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].loglog(mesh_sizes, vertical_errors, "o-", label="u(0.5, y) diff")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Mesh size h")
    axes[0].set_ylabel("RMS difference to reference")
    axes[0].set_title(f"Vertical Centerline, Re = {TARGET_REYNOLDS}")
    axes[0].grid(True, which="both", linestyle="dashed", alpha=0.5)

    axes[1].loglog(mesh_sizes, horizontal_errors, "o-", color="tab:orange", label="v(x, 0.5) diff")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("Mesh size h")
    axes[1].set_ylabel("RMS difference to reference")
    axes[1].set_title(f"Horizontal Centerline, Re = {TARGET_REYNOLDS}")
    axes[1].grid(True, which="both", linestyle="dashed", alpha=0.5)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    parameters = read_parameters("cavity_solver_config.txt")

    reference_parameters = make_parameters_with_mesh(
        parameters,
        REFERENCE_MESH_CELLS,
        sample_points=max(parameters.sample_points, 81),
        time_step=CONVERGENCE_TIME_STEP,
    )
    _, reference_metadata, _, reference_fields = solve_case(
        reference_parameters,
        TARGET_REYNOLDS,
        initial_guess=None,
        output_dir=RESULTS_DIR,
        save_frames=False,
    )
    reference_lines = sample_velocity_centerlines(reference_fields["velocity"], CENTERLINE_POINTS)
    if reference_lines is None:
        raise RuntimeError("Failed to sample reference centerlines.")
    reference_center_u, reference_center_v = center_velocity_from_lines(reference_lines)

    rows = []
    cases = {}
    for mesh_cells in MESH_LEVELS:
        case_parameters = make_parameters_with_mesh(
            parameters,
            mesh_cells,
            sample_points=max(parameters.sample_points, 81),
            time_step=CONVERGENCE_TIME_STEP,
        )
        _, metadata, _, fields = solve_case(
            case_parameters,
            TARGET_REYNOLDS,
            initial_guess=None,
            output_dir=RESULTS_DIR,
            save_frames=False,
        )
        lines = sample_velocity_centerlines(fields["velocity"], CENTERLINE_POINTS)
        if lines is None:
            raise RuntimeError(f"Failed to sample centerlines for mesh {mesh_cells}.")

        center_u, center_v = center_velocity_from_lines(lines)
        mean_newton_iterations = float(
            np.mean([step["newton_iterations"] for step in metadata["iteration_history"]])
        )
        row = {
            "mesh_cells": mesh_cells,
            "mesh_size_h": 1.0 / mesh_cells,
            "ndofs": metadata["ndofs"],
            "reynolds_number": TARGET_REYNOLDS,
            "mean_newton_iterations": mean_newton_iterations,
            "center_u": center_u,
            "center_v": center_v,
            "reference_center_u": reference_center_u,
            "reference_center_v": reference_center_v,
            "vertical_centerline_diff_to_reference": line_l2_difference(
                lines["vertical_u"], reference_lines["vertical_u"]
            ),
            "horizontal_centerline_diff_to_reference": line_l2_difference(
                lines["horizontal_v"], reference_lines["horizontal_v"]
            ),
            "vertical_observed_order": "",
            "horizontal_observed_order": "",
            "centerline_points": CENTERLINE_POINTS,
            "time_step": metadata["time_step"],
            "final_time": metadata["final_time"],
        }
        rows.append(row)
        cases[f"mesh_{mesh_cells}"] = {
            "metadata": metadata,
            "vertical_y": reference_lines["vertical_y"].tolist(),
            "vertical_u": lines["vertical_u"].tolist(),
            "horizontal_x": reference_lines["horizontal_x"].tolist(),
            "horizontal_v": lines["horizontal_v"].tolist(),
            "row": row,
        }

    for index in range(len(rows) - 1):
        coarse = rows[index]
        fine = rows[index + 1]
        h_coarse = coarse["mesh_size_h"]
        h_fine = fine["mesh_size_h"]
        coarse["vertical_observed_order"] = np.log(
            coarse["vertical_centerline_diff_to_reference"] / fine["vertical_centerline_diff_to_reference"]
        ) / np.log(h_coarse / h_fine)
        coarse["horizontal_observed_order"] = np.log(
            coarse["horizontal_centerline_diff_to_reference"] / fine["horizontal_centerline_diff_to_reference"]
        ) / np.log(h_coarse / h_fine)

    csv_path = write_csv(rows)
    figure_path = write_plot(rows)
    summary_path = RESULTS_DIR / "convergence_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "target_reynolds": TARGET_REYNOLDS,
                "mesh_levels": MESH_LEVELS,
                "reference_mesh_cells": REFERENCE_MESH_CELLS,
                "convergence_time_step": CONVERGENCE_TIME_STEP,
                "centerline_points": CENTERLINE_POINTS,
                "csv_file": str(csv_path),
                "figure_file": str(figure_path),
                "cases": cases,
            },
            stream,
            indent=2,
        )

    print("mesh   h         dofs    mean Newton it   center_u      center_v      vert diff     horiz diff")
    for row in rows:
        print(
            f"{row['mesh_cells']:4d} {row['mesh_size_h']:.5f} {row['ndofs']:7d} "
            f"{row['mean_newton_iterations']:16.3f} {row['center_u']:.8e} {row['center_v']:.8e} "
            f"{row['vertical_centerline_diff_to_reference']:.3e} {row['horizontal_centerline_diff_to_reference']:.3e}"
        )


if __name__ == "__main__":
    main()
