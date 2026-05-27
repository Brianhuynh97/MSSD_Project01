from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import Any, cast

from cavity_boundary_conditions import top_boundary, wall_boundary
from cavity_solution_sampling import configure_cache_dirs, sample_functions_on_grid
from cavity_case_io import read_parameters, write_case_data, write_case_frame


def log_progress(domain, message):
    if domain.comm.rank == 0:
        print(message, flush=True)


def build_domain_and_spaces(parameters):
    from basix.ufl import element, mixed_element
    from dolfinx import fem, mesh
    from mpi4py import MPI

    domain = mesh.create_unit_square(MPI.COMM_WORLD, parameters.mesh_cells, parameters.mesh_cells)
    velocity_element = element("Lagrange", domain.basix_cell(), 2, shape=(2,))
    pressure_element = element("Lagrange", domain.basix_cell(), 1)
    mixed_space = fem.functionspace(domain, mixed_element([velocity_element, pressure_element]))
    velocity_space, _ = mixed_space.sub(0).collapse()
    pressure_space, _ = mixed_space.sub(1).collapse()
    return domain, mixed_space, velocity_space, pressure_space


def build_state_functions(mixed_space, initial_guess=None):
    from dolfinx import fem

    current_state = cast(Any, fem.Function(mixed_space))
    previous_state = cast(Any, fem.Function(mixed_space))
    if initial_guess is not None:
        previous_state.x.array[:] = initial_guess
        current_state.x.array[:] = initial_guess
    return current_state, previous_state


def build_boundary_conditions(domain, mixed_space, velocity_space, pressure_space, parameters):
    import numpy as np
    from dolfinx import fem, mesh

    lid_velocity = cast(Any, fem.Function(velocity_space))
    lid_velocity.interpolate(
        lambda x: np.vstack((np.full(x.shape[1], parameters.lid_velocity), np.zeros(x.shape[1])))
    )
    zero_velocity = cast(Any, fem.Function(velocity_space))
    zero_velocity.x.array[:] = 0.0
    zero_pressure = cast(Any, fem.Function(pressure_space))
    zero_pressure.x.array[:] = 0.0

    facet_dim = domain.topology.dim - 1
    lid_facets = mesh.locate_entities_boundary(domain, facet_dim, top_boundary)
    wall_facets = mesh.locate_entities_boundary(domain, facet_dim, wall_boundary)
    lid_dofs = fem.locate_dofs_topological((mixed_space.sub(0), velocity_space), facet_dim, lid_facets)
    wall_dofs = fem.locate_dofs_topological((mixed_space.sub(0), velocity_space), facet_dim, wall_facets)
    pressure_dofs = fem.locate_dofs_geometrical(
        (mixed_space.sub(1), pressure_space),
        lambda x: np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)),
    )
    dirichlet_bc = cast(Any, fem.dirichletbc)
    return [
        dirichlet_bc(cast(Any, lid_velocity), lid_dofs, mixed_space.sub(0)),
        dirichlet_bc(cast(Any, zero_velocity), wall_dofs, mixed_space.sub(0)),
        dirichlet_bc(cast(Any, zero_pressure), pressure_dofs, mixed_space.sub(1)),
    ]


def build_residual_and_jacobian(domain, mixed_space, current_state, previous_state, parameters, reynolds_number):
    import ufl
    from dolfinx import fem

    state_components = cast(tuple[Any, Any], ufl.split(current_state))
    previous_components = cast(tuple[Any, Any], ufl.split(previous_state))
    test_components = cast(tuple[Any, Any], ufl.TestFunctions(mixed_space))
    u = state_components[0]
    p = state_components[1]
    u_previous = previous_components[0]
    v = test_components[0]
    q = test_components[1]
    dw = ufl.TrialFunction(mixed_space)

    dt = cast(Any, fem.Constant(domain, float(parameters.time_step)))
    nu = cast(Any, fem.Constant(domain, float(parameters.lid_velocity / reynolds_number)))
    pressure_penalty = cast(Any, fem.Constant(domain, 1.0e-8))

    time_term = cast(Any, (1.0 / dt) * ufl.inner(u - u_previous, v) * ufl.dx)
    viscous_term = cast(Any, nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    convection_term = cast(Any, ufl.inner(ufl.grad(u) * u, v) * ufl.dx)
    pressure_term = cast(Any, ufl.inner(p, ufl.div(v)) * ufl.dx)
    continuity_term = cast(Any, ufl.inner(q, ufl.div(u)) * ufl.dx)
    pressure_penalty_term = cast(Any, pressure_penalty * ufl.inner(p, q) * ufl.dx)

    residual = (
        time_term
        + viscous_term
        + convection_term
        - pressure_term
        + continuity_term
        + pressure_penalty_term
    )
    jacobian = ufl.derivative(residual, current_state, dw)
    return residual, jacobian


def build_nonlinear_problem(residual, jacobian, current_state, boundary_conditions, parameters, reynolds_number):
    from dolfinx.fem.petsc import NonlinearProblem

    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": parameters.newton_rtol,
        "snes_atol": parameters.newton_atol,
        "snes_max_it": parameters.newton_max_it,
        "snes_error_if_not_converged": True,
        "ksp_type": parameters.ksp_type,
        "pc_type": parameters.pc_type,
    }
    return NonlinearProblem(
        residual,
        current_state,
        bcs=boundary_conditions,
        J=jacobian,
        petsc_options_prefix=f"cavity_re_{reynolds_number}_",
        petsc_options=petsc_options,
    )


def sample_state(domain, state, sample_points):
    velocity = state.sub(0).collapse()
    pressure = state.sub(1).collapse()
    sampled = sample_functions_on_grid(domain, velocity, pressure, sample_points)
    return sampled, velocity, pressure


def write_frame_if_needed(frame_dir, frame_index, time_value, sampled):
    if sampled is None:
        return frame_index
    x, y, u_grid, v_grid, p_grid = sampled
    write_case_frame(frame_dir, frame_index, time_value, x, y, u_grid, v_grid, p_grid)
    return frame_index + 1


def solve_time_steps(problem, domain, current_state, previous_state, parameters, reynolds_number, output_dir, save_frames):
    current_time = 0.0
    steps = int(round(parameters.final_time / parameters.time_step))
    iteration_history = []
    frame_dir = output_dir / f"re_{reynolds_number:05d}_frames" if output_dir is not None else None
    frame_index = 0
    progress_stride = max(1, steps // 20)

    sampled, _, _ = sample_state(domain, previous_state, parameters.sample_points)
    if save_frames and frame_dir is not None:
        frame_index = write_frame_if_needed(frame_dir, frame_index, current_time, sampled)

    for step in range(steps):
        current_state.x.array[:] = previous_state.x.array
        problem.solve()
        converged_reason = problem.solver.getConvergedReason()
        iterations = problem.solver.getIterationNumber()
        current_state.x.scatter_forward()
        iteration_history.append(
            {"step": step + 1, "newton_iterations": int(iterations), "converged_reason": int(converged_reason)}
        )
        if converged_reason <= 0:
            raise RuntimeError(f"SNES solver failed for Re={reynolds_number} at step {step + 1}.")
        previous_state.x.array[:] = current_state.x.array
        previous_state.x.scatter_forward()
        current_time += parameters.time_step

        if step == 0 or (step + 1) % progress_stride == 0 or step + 1 == steps:
            log_progress(
                domain,
                (
                    f"[Re={reynolds_number}] step {step + 1}/{steps} "
                    f"t={current_time:.3f} Newton it={int(iterations)}"
                ),
            )

        if save_frames and frame_dir is not None and ((step + 1) % parameters.frame_stride == 0 or step + 1 == steps):
            sampled, _, _ = sample_state(domain, current_state, parameters.sample_points)
            frame_index = write_frame_if_needed(frame_dir, frame_index, current_time, sampled)

    return current_time, iteration_history


def solve_case(parameters, reynolds_number, initial_guess=None, output_dir: Path | None = None, save_frames: bool = True):
    domain, mixed_space, velocity_space, pressure_space = build_domain_and_spaces(parameters)
    log_progress(
        domain,
        (
            f"Starting Re={reynolds_number} "
            f"(mesh_cells={parameters.mesh_cells}, dt={parameters.time_step}, Tf={parameters.final_time})"
        ),
    )
    current_state, previous_state = build_state_functions(mixed_space, initial_guess)
    boundary_conditions = build_boundary_conditions(
        domain, mixed_space, velocity_space, pressure_space, parameters
    )
    residual, jacobian = build_residual_and_jacobian(
        domain, mixed_space, current_state, previous_state, parameters, reynolds_number
    )
    problem = build_nonlinear_problem(
        residual, jacobian, current_state, boundary_conditions, parameters, reynolds_number
    )

    current_time, iteration_history = solve_time_steps(
        problem,
        domain,
        current_state,
        previous_state,
        parameters,
        reynolds_number,
        output_dir,
        save_frames,
    )

    sampled, velocity, pressure = sample_state(domain, current_state, parameters.sample_points)
    ndofs = mixed_space.dofmap.index_map.size_global * mixed_space.dofmap.index_map_bs
    metadata = {
        "reynolds_number": reynolds_number,
        "lid_velocity": parameters.lid_velocity,
        "kinematic_viscosity": parameters.lid_velocity / reynolds_number,
        "time_step": parameters.time_step,
        "final_time": current_time,
        "mesh_cells": parameters.mesh_cells,
        "sample_points": parameters.sample_points,
        "ndofs": int(ndofs),
        "iteration_history": iteration_history,
    }
    fields = {"domain": domain, "velocity": velocity, "pressure": pressure}
    log_progress(domain, f"Finished Re={reynolds_number} at t={current_time:.3f}")
    return sampled, metadata, current_state.x.array.copy(), fields


def main():
    project_root = Path(__file__).resolve().parent
    configure_cache_dirs(project_root)
    parameters = read_parameters("cavity_solver_config.txt")
    output_dir = project_root / "out"
    output_dir.mkdir(exist_ok=True)

    previous_solution = None
    for reynolds_number in parameters.reynolds_numbers:
        sampled, metadata, previous_solution, _ = solve_case(
            parameters,
            reynolds_number,
            initial_guess=previous_solution,
            output_dir=output_dir,
            save_frames=True,
        )
        if sampled is not None:
            x, y, u, v, p = sampled
            write_case_data(output_dir, reynolds_number, x, y, u, v, p, metadata)


def make_parameters_with_mesh(
    parameters,
    mesh_cells: int,
    sample_points: int | None = None,
    time_step: float | None = None,
):
    return replace(
        parameters,
        mesh_cells=mesh_cells,
        sample_points=sample_points if sample_points is not None else parameters.sample_points,
        time_step=time_step if time_step is not None else parameters.time_step,
    )


if __name__ == "__main__":
    main()
