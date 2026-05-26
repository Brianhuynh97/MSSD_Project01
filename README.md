# Lid-Driven Cavity with FEniCSx

This project solves the 2D lid-driven cavity problem with finite elements using:

- Taylor-Hood function spaces
- UFL residual and Jacobian forms
- Dirichlet boundary conditions
- PETSc-backed Newton solves through FEniCSx

The workflow is built around multiple Reynolds numbers so the output figures
look like the cavity comparisons you showed.

## Files

- [solve_lid_driven_cavity.py](/Users/brianhuynh/Project01_MSSD/solve_lid_driven_cavity.py:1)
  Runs the cavity solves for all Reynolds numbers listed in `cavity_solver_config.txt`.

- [cavity_boundary_conditions.py](/Users/brianhuynh/Project01_MSSD/cavity_boundary_conditions.py:1)
  Boundary locator functions for the top lid and no-slip walls.

- [cavity_solution_sampling.py](/Users/brianhuynh/Project01_MSSD/cavity_solution_sampling.py:1)
  Sampling utilities and cache-directory helpers used by the FEM workflow.

- [cavity_case_io.py](/Users/brianhuynh/Project01_MSSD/cavity_case_io.py:1)
  Parameter parsing and per-case output writing.

- [plot_cavity_case_fields.py](/Users/brianhuynh/Project01_MSSD/plot_cavity_case_fields.py:1)
  Builds the multi-row cavity summary figure with velocity magnitude, quivers,
  and pressure.

- [plot_cavity_ghia_comparison.py](/Users/brianhuynh/Project01_MSSD/plot_cavity_ghia_comparison.py:1)
  Compares the `Re=100` centerline profiles against embedded Ghia reference
  data.

- [cavity_solver_config.txt](/Users/brianhuynh/Project01_MSSD/cavity_solver_config.txt:1)
  Controls mesh size, time step, final time, Reynolds numbers, Newton
  tolerances, PETSc linear solver type, and plot density.

## Mathematical Model

The code solves the incompressible Navier-Stokes equations on the unit square:

```text
du/dt + (u . grad) u - nu Delta u + grad p = 0
div u = 0
```

with lid-driven cavity boundary conditions:

- top boundary: `u = (1, 0)`
- left, right, and bottom boundaries: `u = (0, 0)`

Because the cavity problem has no prescribed pressure boundary, one pressure
degree of freedom is pinned at the origin to remove the null space.

## Discretization

The solver uses:

- mesh: unit-square triangular mesh
- velocity space: quadratic Lagrange (`P2`)
- pressure space: linear Lagrange (`P1`)

The time discretization is backward Euler and the nonlinear convective term is
handled through a Newton solve at each time step.

## Residual and Jacobian

For each time step, the unknown mixed function `w = (u, p)` satisfies a UFL
residual of the form:

```text
(1/dt) (u - u_n, v)
+ nu (grad u, grad v)
+ ((grad u) u, v)
- (p, div v)
+ (q, div u)
+ eps (p, q)
```

and the Jacobian is obtained with `ufl.derivative`.

## Environment

This project expects a FEniCSx environment with:

- `dolfinx`
- `basix`
- `ufl`
- `petsc4py`
- `mpi4py`
- `numpy`
- `matplotlib`

The simplest route is a conda-forge FEniCSx environment.

## Run

Solve all configured Reynolds numbers:

```bash
python solve_lid_driven_cavity.py
```

Generate the summary plots:

```bash
python plot_cavity_case_fields.py
python plot_cavity_ghia_comparison.py
```

Run the mesh convergence study:

```bash
python run_cavity_convergence.py
```

## Outputs

Per Reynolds number, `solve_lid_driven_cavity.py` writes:

- `out/re_00010.npz`
- `out/re_00010.json`
- `out/re_00100.npz`
- `out/re_00100.json`
- `out/re_01000.npz`
- `out/re_01000.json`
- `out/re_10000.npz`
- `out/re_10000.json`

The plot scripts write:

- `figures/lid_driven_cavity_re_00010.png`
- `figures/lid_driven_cavity_re_00010.gif`
- `figures/lid_driven_cavity_re_00100.png`
- `figures/lid_driven_cavity_re_00100.gif`
- `figures/lid_driven_cavity_re_01000.png`
- `figures/lid_driven_cavity_re_01000.gif`
- `figures/lid_driven_cavity_re_10000.png`
- `figures/lid_driven_cavity_re_10000.gif`
- `figures/ghia_comparison_re100.png`

The convergence script writes:

- `results/convergence/convergence_summary.csv`
- `results/convergence/convergence_summary.json`
- `figures/convergence/convergence_summary.png`
