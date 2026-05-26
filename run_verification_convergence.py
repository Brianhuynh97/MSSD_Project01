from __future__ import annotations

import csv
import inspect
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cavity_solution_sampling import configure_cache_dirs


MESH_LEVELS = [8, 16, 32, 64]
RESULTS_DIR = Path("results/verification")
FIGURES_DIR = Path("figures/verification")
KINEMATIC_VISCOSITY = 1.0
PRESSURE_PENALTY = 1.0e-6


def manufactured_velocity(x):
    import ufl

    return ufl.as_vector(
        [
            ufl.pi * ufl.sin(ufl.pi * x[0]) ** 2 * ufl.sin(2.0 * ufl.pi * x[1]),
            -ufl.pi * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) ** 2,
        ]
    )


def manufactured_pressure(x):
    return 0.0 * x[0]


def solve_verification_case(nx: int) -> dict[str, float]:
    import ufl
    from basix.ufl import element, mixed_element
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI

    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, nx)
    x = ufl.SpatialCoordinate(domain)

    velocity_element = element("Lagrange", domain.basix_cell(), 2, shape=(2,))
    pressure_element = element("Lagrange", domain.basix_cell(), 1)
    mixed_space = fem.functionspace(domain, mixed_element([velocity_element, pressure_element]))
    velocity_space, _ = mixed_space.sub(0).collapse()
    pressure_space, _ = mixed_space.sub(1).collapse()

    w = ufl.TrialFunction(mixed_space)
    z = ufl.TestFunction(mixed_space)
    u, p = ufl.split(w)
    v, q = ufl.split(z)

    u_exact = manufactured_velocity(x)
    p_exact = manufactured_pressure(x)
    forcing = -KINEMATIC_VISCOSITY * ufl.div(ufl.grad(u_exact))

    a = (
        KINEMATIC_VISCOSITY * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(q, ufl.div(u)) * ufl.dx
        + PRESSURE_PENALTY * ufl.inner(p, q) * ufl.dx
    )
    L = ufl.inner(forcing, v) * ufl.dx

    u_exact_function = fem.Function(velocity_space)
    u_exact_function.interpolate(lambda points: np.vstack(
        [
            math.pi * np.sin(math.pi * points[0]) ** 2 * np.sin(2.0 * math.pi * points[1]),
            -math.pi * np.sin(2.0 * math.pi * points[0]) * np.sin(math.pi * points[1]) ** 2,
        ]
    ))
    p_exact_function = fem.Function(pressure_space)
    p_exact_function.x.array[:] = 0.0

    facet_dim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, facet_dim, lambda points: np.full(points.shape[1], True))
    velocity_dofs = fem.locate_dofs_topological((mixed_space.sub(0), velocity_space), facet_dim, boundary_facets)
    boundary_conditions = [
        fem.dirichletbc(u_exact_function, velocity_dofs, mixed_space.sub(0)),
    ]

    problem_kwargs = {
        "bcs": boundary_conditions,
        "petsc_options": {
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    }
    if "petsc_options_prefix" in inspect.signature(LinearProblem).parameters:
        problem_kwargs["petsc_options_prefix"] = f"verification_{nx}_"
    problem = LinearProblem(a, L, **problem_kwargs)
    problem.solver.setConvergenceHistory()
    solution = problem.solve()
    solution.x.scatter_forward()

    velocity_h = solution.sub(0).collapse()
    pressure_h = solution.sub(1).collapse()

    velocity_l2_error_local = fem.assemble_scalar(
        fem.form(ufl.inner(velocity_h - u_exact, velocity_h - u_exact) * ufl.dx)
    )
    velocity_h1_error_local = fem.assemble_scalar(
        fem.form(ufl.inner(ufl.grad(velocity_h - u_exact), ufl.grad(velocity_h - u_exact)) * ufl.dx)
    )
    pressure_l2_error_local = fem.assemble_scalar(
        fem.form(ufl.inner(pressure_h - p_exact, pressure_h - p_exact) * ufl.dx)
    )
    divergence_l2_error_local = fem.assemble_scalar(
        fem.form(ufl.inner(ufl.div(velocity_h), ufl.div(velocity_h)) * ufl.dx)
    )

    velocity_l2_error = math.sqrt(domain.comm.allreduce(velocity_l2_error_local, op=MPI.SUM))
    velocity_h1_error = math.sqrt(domain.comm.allreduce(velocity_h1_error_local, op=MPI.SUM))
    pressure_l2_error = math.sqrt(domain.comm.allreduce(pressure_l2_error_local, op=MPI.SUM))
    divergence_l2_error = math.sqrt(domain.comm.allreduce(divergence_l2_error_local, op=MPI.SUM))
    ndofs = mixed_space.dofmap.index_map.size_global * mixed_space.dofmap.index_map_bs

    return {
        "mesh_cells": nx,
        "mesh_size_h": 1.0 / nx,
        "ndofs": int(ndofs),
        "linear_iterations": int(problem.solver.getIterationNumber()),
        "velocity_l2_error": velocity_l2_error,
        "velocity_h1_seminorm_error": velocity_h1_error,
        "pressure_l2_error": pressure_l2_error,
        "divergence_l2_error": divergence_l2_error,
        "velocity_l2_rate": "",
        "velocity_h1_rate": "",
        "pressure_l2_rate": "",
        "divergence_l2_rate": "",
    }


def add_observed_rates(rows: list[dict[str, float]]) -> None:
    rate_fields = [
        ("velocity_l2_error", "velocity_l2_rate"),
        ("velocity_h1_seminorm_error", "velocity_h1_rate"),
        ("pressure_l2_error", "pressure_l2_rate"),
        ("divergence_l2_error", "divergence_l2_rate"),
    ]
    for coarse, fine in zip(rows[:-1], rows[1:]):
        for error_key, rate_key in rate_fields:
            if (
                coarse[error_key] > 0.0
                and fine[error_key] > 0.0
                and math.isfinite(coarse[error_key])
                and math.isfinite(fine[error_key])
            ):
                coarse[rate_key] = math.log(coarse[error_key] / fine[error_key]) / math.log(
                    coarse["mesh_size_h"] / fine["mesh_size_h"]
                )
            else:
                coarse[rate_key] = ""


def write_csv(rows: list[dict[str, float]]) -> Path:
    csv_path = RESULTS_DIR / "verification_convergence.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "mesh_cells",
                "mesh_size_h",
                "ndofs",
                "linear_iterations",
                "velocity_l2_error",
                "velocity_h1_seminorm_error",
                "pressure_l2_error",
                "divergence_l2_error",
                "velocity_l2_rate",
                "velocity_h1_rate",
                "pressure_l2_rate",
                "divergence_l2_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def write_plot(rows: list[dict[str, float]]) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURES_DIR / "verification_convergence.png"
    h_values = np.array([row["mesh_size_h"] for row in rows])
    tiny = np.finfo(float).tiny
    velocity_l2_values = np.maximum([row["velocity_l2_error"] for row in rows], tiny)
    velocity_h1_values = np.maximum([row["velocity_h1_seminorm_error"] for row in rows], tiny)
    pressure_l2_values = np.maximum([row["pressure_l2_error"] for row in rows], tiny)
    divergence_values = np.maximum([row["divergence_l2_error"] for row in rows], tiny)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(h_values, velocity_l2_values, "o-", label=r"$\|u-u_h\|_{L^2}$")
    ax.loglog(
        h_values,
        velocity_h1_values,
        "s-",
        label=r"$|u-u_h|_{H^1}$",
    )
    ax.loglog(h_values, pressure_l2_values, "^-", label=r"$\|p-p_h\|_{L^2}$")
    ax.loglog(
        h_values,
        divergence_values,
        "d-",
        label=r"$\|\nabla\cdot u_h\|_{L^2}$",
    )

    reference_h = np.array([h_values[-2], h_values[-1]])
    ax.loglog(
        reference_h,
        0.8 * rows[-2]["velocity_h1_seminorm_error"] * (reference_h / reference_h[0]) ** 2,
        "k--",
        linewidth=1.0,
        label=r"$\mathcal{O}(h^2)$",
    )
    ax.loglog(
        reference_h,
        1.2 * rows[-2]["velocity_l2_error"] * (reference_h / reference_h[0]) ** 3,
        "k:",
        linewidth=1.0,
        label=r"$\mathcal{O}(h^3)$",
    )

    ax.invert_xaxis()
    ax.set_xlabel("Mesh size h = 1 / nx")
    ax.set_ylabel("Error norm")
    ax.set_title("Manufactured Stokes Verification Convergence")
    ax.grid(True, which="both", linestyle="dashed", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def write_summary(rows: list[dict[str, float]], csv_path: Path, figure_path: Path) -> Path:
    finite_velocity_l2_rates = [row["velocity_l2_rate"] for row in rows if row["velocity_l2_rate"] != ""]
    finite_velocity_h1_rates = [row["velocity_h1_rate"] for row in rows if row["velocity_h1_rate"] != ""]
    finite_pressure_rates = [row["pressure_l2_rate"] for row in rows if row["pressure_l2_rate"] != ""]
    finite_divergence_rates = [row["divergence_l2_rate"] for row in rows if row["divergence_l2_rate"] != ""]
    summary_path = RESULTS_DIR / "verification_convergence.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "problem": "manufactured_steady_stokes",
                "mesh_levels": MESH_LEVELS,
                "mesh_size_definition": "h = 1 / nx for a uniform unit-square mesh with nx subdivisions per side",
                "elements": {
                    "velocity": "P2 Lagrange",
                    "pressure": "P1 Lagrange",
                },
                "expected_rates": {
                    "velocity_l2_error": 3.0,
                    "velocity_h1_seminorm_error": 2.0,
                    "pressure_l2_error": 2.0,
                },
                "observed_average_rates": {
                    "velocity_l2_error": float(np.mean(finite_velocity_l2_rates)),
                    "velocity_h1_seminorm_error": float(np.mean(finite_velocity_h1_rates)),
                    "pressure_l2_error": float(np.mean(finite_pressure_rates)),
                    "divergence_l2_error": float(np.mean(finite_divergence_rates)),
                },
                "rate_interpretation": {
                    "velocity_l2_error": "Matches the expected cubic L2 convergence for the P2 velocity space on a smooth Stokes problem.",
                    "velocity_h1_seminorm_error": "Matches the expected quadratic H1-seminorm convergence for the P2 velocity space.",
                    "pressure_l2_error": "Observed pressure convergence is higher than the generic P1 expectation because the manufactured solution uses p = 0 and the small pressure penalty removes the nullspace without changing the exact solution.",
                },
                "limitations": [
                    "This is a steady manufactured Stokes verification problem, not the nonlinear transient cavity benchmark.",
                    "The divergence norm is included as a diagnostic quantity; its observed rate is not the primary theoretical target here.",
                    "The pressure convergence behavior is specific to this manufactured zero-pressure verification case and should not be interpreted as a universal rate for all cavity or Navier-Stokes runs.",
                ],
                "csv_file": str(csv_path),
                "figure_file": str(figure_path),
                "rows": rows,
            },
            stream,
            indent=2,
        )
    return summary_path


def main():
    project_root = Path(__file__).resolve().parent
    configure_cache_dirs(project_root)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = [solve_verification_case(nx) for nx in MESH_LEVELS]
    add_observed_rates(rows)
    csv_path = write_csv(rows)
    figure_path = write_plot(rows)
    summary_path = write_summary(rows, csv_path, figure_path)

    print("nx    h         dofs    iters   ||u-uh||L2   |u-uh|H1   ||p-ph||L2   ||div uh||L2")
    for row in rows:
        print(
            f"{row['mesh_cells']:4d} {row['mesh_size_h']:.5f} {row['ndofs']:7d} {row['linear_iterations']:6d} "
            f"{row['velocity_l2_error']:.3e} {row['velocity_h1_seminorm_error']:.3e} "
            f"{row['pressure_l2_error']:.3e} {row['divergence_l2_error']:.3e}"
        )
    print(f"Wrote {csv_path}")
    print(f"Wrote {figure_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
