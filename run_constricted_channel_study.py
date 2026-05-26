from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from cavity_solution_sampling import configure_cache_dirs
from generate_constricted_channel_mesh import (
    CHANNEL_HEIGHT,
    CHANNEL_LENGTH,
    INLET_TAG,
    OUTLET_TAG,
    VARIANTS,
    WALL_TAG,
    generate_all_meshes,
)


RESULTS_DIR = Path("results/physical_study")
FIGURES_DIR = Path("figures/physical_study")
MEAN_INLET_SPEED = 1.0
KINEMATIC_VISCOSITY = 0.02
PRESSURE_PENALTY = 1.0e-8
GRID_SHAPE = (220, 70)
SampleGrid: TypeAlias = dict[str, np.ndarray]


@dataclass(frozen=True)
class StudyResult:
    variant: str
    notch_radius: float
    mesh_file: str
    num_cells: int
    num_vertices: int
    ndofs: int
    linear_iterations: int
    pressure_drop: float
    max_speed: float
    outlet_flow_rate: float


def read_tagged_mesh(mesh_file: Path):
    from dolfinx.io import gmsh
    from mpi4py import MPI

    mesh_data = gmsh.read_from_msh(mesh_file, MPI.COMM_WORLD, 0, gdim=2)
    return mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags


def build_stokes_problem(domain, facet_tags, variant_name: str):
    import ufl
    from basix.ufl import element, mixed_element
    from dolfinx import fem
    from dolfinx.fem.petsc import LinearProblem

    velocity_element = element("Lagrange", domain.basix_cell(), 2, shape=(2,))
    pressure_element = element("Lagrange", domain.basix_cell(), 1)
    mixed_space = fem.functionspace(domain, mixed_element([velocity_element, pressure_element]))
    velocity_space, _ = mixed_space.sub(0).collapse()

    trial = ufl.TrialFunction(mixed_space)
    test = ufl.TestFunction(mixed_space)
    trial_components = cast(tuple[Any, Any], ufl.split(trial))
    test_components = cast(tuple[Any, Any], ufl.split(test))
    u = trial_components[0]
    p = trial_components[1]
    v = test_components[0]
    q = test_components[1]

    x = ufl.SpatialCoordinate(domain)
    inlet_velocity = cast(Any, fem.Function(velocity_space))
    inlet_velocity.interpolate(
        lambda points: np.vstack(
            [
                6.0
                * MEAN_INLET_SPEED
                * points[1]
                * (CHANNEL_HEIGHT - points[1])
                / CHANNEL_HEIGHT**2,
                np.zeros(points.shape[1]),
            ]
        )
    )
    zero_velocity = cast(Any, fem.Function(velocity_space))
    zero_velocity.x.array[:] = 0.0

    facet_dim = domain.topology.dim - 1
    inlet_facets = facet_tags.find(INLET_TAG)
    wall_facets = facet_tags.find(WALL_TAG)
    inlet_dofs = fem.locate_dofs_topological((mixed_space.sub(0), velocity_space), facet_dim, inlet_facets)
    wall_dofs = fem.locate_dofs_topological((mixed_space.sub(0), velocity_space), facet_dim, wall_facets)

    dirichlet_bc = cast(Any, fem.dirichletbc)

    bcs = [
    dirichlet_bc(cast(Any, inlet_velocity), inlet_dofs, mixed_space.sub(0)),
    dirichlet_bc(cast(Any, zero_velocity), wall_dofs, mixed_space.sub(0)),
]
    kinematic_viscosity = ufl.as_ufl(KINEMATIC_VISCOSITY)
    pressure_penalty = ufl.as_ufl(PRESSURE_PENALTY)
    viscous_term = cast(Any, kinematic_viscosity * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    pressure_term = cast(Any, ufl.inner(p, ufl.div(v)) * ufl.dx)
    continuity_term = cast(Any, ufl.inner(q, ufl.div(u)) * ufl.dx)
    pressure_penalty_term = cast(Any, pressure_penalty * ufl.inner(p, q) * ufl.dx)
    a = viscous_term - pressure_term + continuity_term + pressure_penalty_term
    zero_force = fem.Constant(domain, np.array((0.0, 0.0), dtype=np.float64))
    L = cast(Any, ufl.inner(zero_force, v) * ufl.dx)

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix=f"constricted_channel_{variant_name}_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    return problem, mixed_space


def solve_variant(mesh_file: Path, variant_name: str, notch_radius: float):
    import ufl
    from dolfinx import fem, mesh

    domain, cell_tags, facet_tags = read_tagged_mesh(mesh_file)
    problem, mixed_space = build_stokes_problem(domain, facet_tags, variant_name)
    #solution = problem.solve()
    solution = cast(Any, problem.solve())
    solution.x.scatter_forward()

    velocity = solution.sub(0).collapse()
    pressure = solution.sub(1).collapse()

    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    n = ufl.FacetNormal(domain)
    inlet_length_form = cast(Any, fem.form(1.0 * ds(INLET_TAG)))
    outlet_length_form = cast(Any, fem.form(1.0 * ds(OUTLET_TAG)))
    inlet_pressure_form = cast(Any, fem.form(pressure * ds(INLET_TAG)))
    outlet_pressure_form = cast(Any, fem.form(pressure * ds(OUTLET_TAG)))
    outlet_flux_form = cast(Any, fem.form(ufl.dot(velocity, n) * ds(OUTLET_TAG)))

    inlet_length = fem.assemble_scalar(inlet_length_form)
    outlet_length = fem.assemble_scalar(outlet_length_form)
    inlet_pressure_integral = fem.assemble_scalar(inlet_pressure_form)
    outlet_pressure_integral = fem.assemble_scalar(outlet_pressure_form)
    outlet_flux = fem.assemble_scalar(outlet_flux_form)

    pressure_drop = float(np.real(inlet_pressure_integral / inlet_length - outlet_pressure_integral / outlet_length))
    outlet_flow_rate = float(np.real(outlet_flux))

    sample = sample_velocity_pressure(domain, velocity, pressure, GRID_SHAPE)
    if sample is None:
        max_speed = float("nan")
    else:
        speed_sample = np.sqrt(sample["u"] ** 2 + sample["v"] ** 2)
        max_speed = float(np.nanmax(speed_sample))

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    cell_vertices = domain.topology.connectivity(tdim, 0).array.reshape(-1, 3)
    points = domain.geometry.x[:, :2]
    ndofs = mixed_space.dofmap.index_map.size_global * mixed_space.dofmap.index_map_bs

    result = StudyResult(
        variant=variant_name,
        notch_radius=notch_radius,
        mesh_file=str(mesh_file),
        num_cells=cell_vertices.shape[0],
        num_vertices=points.shape[0],
        ndofs=int(ndofs),
        linear_iterations=int(problem.solver.getIterationNumber()),
        pressure_drop=pressure_drop,
        max_speed=max_speed,
        outlet_flow_rate=outlet_flow_rate,
    )
    return result, domain, cell_tags, facet_tags, velocity, pressure, sample


def sample_velocity_pressure(domain, velocity, pressure, grid_shape: tuple[int, int]) -> SampleGrid | None:
    from dolfinx import geometry

    nx, ny = grid_shape
    xs = np.linspace(0.0, CHANNEL_LENGTH, num=int(nx))
    ys = np.linspace(0.0, CHANNEL_HEIGHT, num=int(ny))
    xy = np.array([[x_value, y_value] for y_value in ys for x_value in xs], dtype=np.float64)
    points = np.column_stack([xy, np.zeros(len(xy), dtype=np.float64)])
    tree = geometry.bb_tree(domain, domain.topology.dim)
    collision_index = np.int32(0)

    local_points: list[np.ndarray] = []
    local_cells: list[int] = []
    local_indices: list[int] = []
    for index, point in enumerate(points):
        candidates = geometry.compute_collisions_points(tree, point[None, :])
        colliding = geometry.compute_colliding_cells(domain, candidates, point[None, :])
        links = colliding.links(collision_index)
        if len(links) > 0:
            local_points.append(point)
            local_cells.append(int(links[0]))
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

    u_grid = np.full((ny, nx), np.nan)
    v_grid = np.full((ny, nx), np.nan)
    p_grid = np.full((ny, nx), np.nan)
    for (indices, values_u), (_, values_p) in zip(gathered_velocity, gathered_pressure):
        if values_u is None or values_p is None:
            continue
        for local_pos, global_index in enumerate(indices):
            j = global_index // nx
            i = global_index % nx
            u_grid[j, i] = values_u[local_pos, 0]
            v_grid[j, i] = values_u[local_pos, 1]
            p_grid[j, i] = values_p[local_pos, 0]

    x_grid, y_grid = np.meshgrid(xs, ys)
    return {"x": x_grid, "y": y_grid, "u": u_grid, "v": v_grid, "p": p_grid}


def plot_mesh_and_tags(domain, facet_tags, result: StudyResult):
    from dolfinx import mesh

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    cells = domain.topology.connectivity(tdim, 0).array.reshape(-1, 3)
    points = domain.geometry.x[:, :2]
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], cells)

    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.triplot(triangulation, color="0.75", linewidth=0.4)

    facet_dim = tdim - 1
    for tag, color, label in [
        (INLET_TAG, "tab:green", "inlet"),
        (OUTLET_TAG, "tab:red", "outlet"),
        (WALL_TAG, "tab:blue", "walls"),
    ]:
        facets = facet_tags.find(tag)
        if len(facets) == 0:
            continue
        midpoints = mesh.compute_midpoints(domain, facet_dim, facets)
        ax.scatter(midpoints[:, 0], midpoints[:, 1], s=8, color=color, label=label)

    ax.set_aspect("equal")
    ax.set_title(f"Mesh and Boundary Tags: {result.variant}")
    ax.legend(loc="upper right")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{result.variant}_mesh_tags.png", dpi=150)
    plt.close(fig)


def plot_fields(sample: SampleGrid, result: StudyResult):
    speed = np.sqrt(sample["u"] ** 2 + sample["v"] ** 2)
    centered_pressure = sample["p"] - np.nanmean(sample["p"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    speed_ax, pressure_ax = axes

    speed_contour = speed_ax.contourf(sample["x"], sample["y"], speed, levels=24, cmap="viridis")
    speed_ax.quiver(
        sample["x"][::4, ::8],
        sample["y"][::4, ::8],
        sample["u"][::4, ::8],
        sample["v"][::4, ::8],
        color="white",
        pivot="mid",
        scale=35,
    )
    speed_ax.set_title(f"Speed, {result.variant}")
    speed_ax.set_aspect("equal")
    fig.colorbar(speed_contour, ax=speed_ax, fraction=0.046, pad=0.02)

    pressure_contour = pressure_ax.contourf(
        sample["x"],
        sample["y"],
        centered_pressure,
        levels=24,
        cmap="coolwarm",
    )
    pressure_ax.set_title(f"Pressure, {result.variant}")
    pressure_ax.set_aspect("equal")
    fig.colorbar(pressure_contour, ax=pressure_ax, fraction=0.046, pad=0.02)

    fig.suptitle(
        f"{result.variant.capitalize()} constriction: Δp={result.pressure_drop:.3f}, max|u|={result.max_speed:.3f}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{result.variant}_fields.png", dpi=150)
    plt.close(fig)


def plot_comparison(results: list[StudyResult]):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    names = [result.variant for result in results]
    pressure_drops = [result.pressure_drop for result in results]
    max_speeds = [result.max_speed for result in results]

    axes[0].bar(names, pressure_drops, color=["tab:green", "tab:red"])
    axes[0].set_title("Pressure Drop")
    axes[0].set_ylabel("Average inlet pressure - average outlet pressure")

    axes[1].bar(names, max_speeds, color=["tab:green", "tab:red"])
    axes[1].set_title("Maximum Speed")
    axes[1].set_ylabel("Max |u| on sampled grid")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "variant_comparison.png", dpi=150)
    plt.close(fig)


def write_outputs(results: list[StudyResult]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "constricted_channel_summary.json"
    csv_path = RESULTS_DIR / "constricted_channel_summary.csv"

    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    better_variant = min(results, key=lambda result: result.pressure_drop)
    summary = {
        "scenario": "steady low-Re channel flow through a symmetric channel constriction",
        "mesh_generation": "Gmsh Python API with gmsh.model.occ",
        "mesh_import": "dolfinx.io.gmsh.read_from_msh",
        "mesh_size_control": "variant-specific global mesh size with local threshold refinement near tagged walls",
        "quantity_of_interest": {
            "primary": "pressure_drop = average inlet pressure - average outlet pressure",
            "secondary": [
                "maximum speed on a reproducible sampling grid",
                "outlet flow rate",
            ],
        },
        "variants": [asdict(result) for result in results],
        "interpretation": {
            "better_variant": better_variant.variant,
            "reason": "Lower pressure drop at the same inlet profile indicates lower hydraulic resistance.",
            "physical_reading": [
                "The severe constriction creates a narrower throat, accelerating the flow and increasing viscous losses.",
                "Pressure gradients concentrate across the throat, while high-speed regions form through the constriction.",
                "The mild constriction performs better for the pressure-drop metric because the channel cross-section remains less restrictive.",
            ],
        },
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)


def main():
    project_root = Path(__file__).resolve().parent
    configure_cache_dirs(project_root)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    mesh_dir = RESULTS_DIR / "meshes"
    mesh_paths = generate_all_meshes(mesh_dir)

    results: list[StudyResult] = []
    for name, variant in VARIANTS.items():
        result, domain, _, facet_tags, _, _, sample = solve_variant(mesh_paths[name], name, variant.notch_radius)
        if domain.comm.rank == 0 and sample is not None:
            plot_mesh_and_tags(domain, facet_tags, result)
            plot_fields(sample, result)
        results.append(result)

    if results:
        write_outputs(results)
        plot_comparison(results)
        print("variant   pressure_drop   max_speed   outlet_flow_rate")
        for result in results:
            print(
                f"{result.variant:7s} {result.pressure_drop:13.6f} {result.max_speed:10.6f} {result.outlet_flow_rate:16.6f}"
            )


if __name__ == "__main__":
    main()
