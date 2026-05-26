# Lid-Driven Cavity with FEniCSx

This project solves the 2D lid-driven cavity problem using the finite element
method in FEniCSx with:

- Taylor–Hood finite elements
- UFL residual and Jacobian forms
- Dirichlet boundary conditions
- PETSc-backed Newton solvers

The workflow is designed for multiple Reynolds numbers so the generated figures
and animations reproduce the classical lid-driven cavity benchmark behavior.

---

# Files

- `solve_lid_driven_cavity.py`  
  Runs the cavity simulations for all Reynolds numbers listed in
  `cavity_solver_config.txt`.

- `cavity_boundary_conditions.py`  
  Boundary locator utilities for the moving lid and no-slip walls.

- `cavity_solution_sampling.py`  
  Sampling utilities and cache-directory helpers used by the FEM workflow.

- `cavity_case_io.py`  
  Parameter parsing and per-case output writing.

- `plot_cavity_case_fields.py`  
  Generates velocity magnitude, streamline, quiver, and pressure plots.

- `plot_cavity_ghia_comparison.py`  
  Compares the `Re=100` centerline velocity profiles against the classical
  Ghia benchmark data.

- `run_cavity_convergence.py`  
  Performs the mesh convergence study.

- `run_verification_convergence.py`  
  Solves a manufactured steady Stokes problem on a sequence of meshes and
  computes exact error norms for verification.

- `generate_stenosis_mesh.py`  
  Builds reproducible Gmsh meshes for a stenosed channel with physical facet
  tags for inlet, outlet, and walls.

- `run_stenosis_study.py`  
  Solves a tagged low-Re channel-flow problem for two stenosis variants and
  compares pressure drop and peak speed.

- `cavity_solver_config.txt`  
  Stores Reynolds numbers, mesh size, time step, solver tolerances,
  PETSc configuration, and plotting settings.

---

## Mathematical Model

The code solves the incompressible Navier–Stokes equations on the unit square:

```math
\frac{\partial \mathbf{u}}{\partial t}
+
(\mathbf{u}\cdot\nabla)\mathbf{u}
-
\nu \Delta \mathbf{u}
+
\nabla p
=
0
```

```math
\nabla \cdot \mathbf{u} = 0
```

where:

- $\mathbf{u}$ is the velocity field
- $p$ is the pressure
- $\nu$ is the kinematic viscosity

---

# Boundary Conditions

The lid-driven cavity boundary conditions are:

## Moving lid (top wall)

$$
\mathbf{u} = (1,0)
$$

## Stationary walls

$$
\mathbf{u} = (0,0)
$$

applied on the left, right, and bottom boundaries.

Because the incompressible Navier–Stokes equations determine pressure only up to
a constant, one pressure degree of freedom is pinned at the origin to remove
the null space.

---

# Finite Element Discretization

The solver uses:

- Unit-square triangular mesh
- Quadratic Lagrange elements (`P2`) for velocity
- Linear Lagrange elements (`P1`) for pressure

This corresponds to the classical Taylor–Hood mixed finite element pair.

The time discretization uses the backward Euler method, while the nonlinear
convective term is solved using Newton iterations.

---

# Weak Formulation

For each time step, the unknown mixed solution

$$
w = (\mathbf{u}, p)
$$

satisfies the residual form:

$$
\begin{aligned}
F(w) =
&\;
\frac{1}{\Delta t}
(\mathbf{u} - \mathbf{u}_n,\mathbf{v})
\\
&+
\nu (\nabla \mathbf{u}, \nabla \mathbf{v})
\\
&+
((\mathbf{u}\cdot\nabla)\mathbf{u}, \mathbf{v})
\\
&-
(p,\nabla\cdot\mathbf{v})
\\
&+
(q,\nabla\cdot\mathbf{u})
\\
&+
\varepsilon (p,q)
\end{aligned}
$$

where:

- $\mathbf{v}$ is the velocity test function
- $q$ is the pressure test function
- $\varepsilon$ is a small pressure stabilization parameter

The Jacobian is obtained automatically using:

```python
ufl.derivative(F, w, dw)
```

---

# Environment

This project requires a FEniCSx environment with:

- `dolfinx`
- `basix`
- `ufl`
- `petsc4py`
- `mpi4py`
- `numpy`
- `matplotlib`

The recommended installation route is a conda-forge FEniCSx environment.

---

# Run

## Solve all Reynolds-number cases

```bash
python solve_lid_driven_cavity.py
```

## Generate cavity plots

```bash
python plot_cavity_case_fields.py
python plot_cavity_ghia_comparison.py
```

## Run convergence study

```bash
python run_cavity_convergence.py
```

## Run verification convergence study

```bash
python run_verification_convergence.py
```

## Run physical stenosis study

```bash
python run_stenosis_study.py
```

---

# Results

## Reynolds Number Cases

## Re = 10

<p align="center">
  <img src="figures/lid_driven_cavity_re_00010.gif" width="700"/>
</p>

## Re = 100

<p align="center">
  <img src="figures/lid_driven_cavity_re_00100.gif" width="700"/>
</p>

## Re = 1000

<p align="center">
  <img src="figures/lid_driven_cavity_re_01000.gif" width="700"/>
</p>

## Re = 10000

<p align="center">
  <img src="figures/lid_driven_cavity_re_10000.gif" width="700"/>
</p>

---

# Output Files

Per Reynolds number, the solver writes:

- `out/re_00010.npz`
- `out/re_00010.json`
- `out/re_00100.npz`
- `out/re_00100.json`
- `out/re_01000.npz`
- `out/re_01000.json`
- `out/re_10000.npz`
- `out/re_10000.json`

Generated figures:

- `figures/lid_driven_cavity_re_00010.png`
- `figures/lid_driven_cavity_re_00010.gif`
- `figures/lid_driven_cavity_re_00100.png`
- `figures/lid_driven_cavity_re_00100.gif`
- `figures/lid_driven_cavity_re_01000.png`
- `figures/lid_driven_cavity_re_01000.gif`
- `figures/lid_driven_cavity_re_10000.png`
- `figures/lid_driven_cavity_re_10000.gif`
- `figures/ghia_comparison_re100.png`

Convergence outputs:

- `results/convergence/convergence_summary.csv`
- `results/convergence/convergence_summary.json`
- `figures/convergence/convergence_summary.png`

Verification outputs:

- `results/verification/verification_convergence.csv`
- `results/verification/verification_convergence.json`
- `figures/verification/verification_convergence.png`

Physical-study outputs:

- `results/physical_study/meshes/stenosis_mild.msh`
- `results/physical_study/meshes/stenosis_severe.msh`
- `results/physical_study/stenosis_summary.csv`
- `results/physical_study/stenosis_summary.json`
- `figures/physical_study/mild_mesh_tags.png`
- `figures/physical_study/mild_fields.png`
- `figures/physical_study/severe_mesh_tags.png`
- `figures/physical_study/severe_fields.png`
- `figures/physical_study/variant_comparison.png`

---

# Verification Study

The project also includes a manufactured steady Stokes verification problem for
the Taylor–Hood `P2/P1` element pair.

## Mesh levels

The verification script uses at least four uniform unit-square meshes:

- `nx = 8`
- `nx = 16`
- `nx = 32`
- `nx = 64`

The mesh-size measure is defined as:

```text
h = 1 / nx
```

for a unit square subdivided uniformly into `nx` cells per coordinate
direction.

## Error norms

The script computes:

- velocity `L2` error: `||u - uh||L2`
- velocity `H1` seminorm error: `|u - uh|H1`
- pressure `L2` error: `||p - ph||L2`
- divergence diagnostic: `||div uh||L2`

## Expected convergence behavior

For a smooth Stokes solution discretized with Taylor–Hood `P2/P1` elements on
quasi-uniform meshes, the standard expectation is:

- `||u - uh||L2 = O(h^3)`
- `|u - uh|H1 = O(h^2)`
- `||p - ph||L2 = O(h^2)`

These are the rates the verification study compares against. The divergence
quantity is included as a diagnostic rather than the main theoretical target.

## Current observed rates

From the current `8 -> 16 -> 32 -> 64` study, the observed rates are:

- velocity `L2`: about `3.00`
- velocity `H1` seminorm: about `1.98`
- pressure `L2`: about `3.60`
- divergence diagnostic: about `1.96`

The velocity rates match the standard Taylor–Hood expectation very closely.
The pressure rate is higher than the generic `P1` expectation because this
manufactured verification case uses an exact zero-pressure field together with
a small pressure penalty to remove the nullspace. That behavior is specific to
this verification setup and should not be treated as a universal cavity-flow
pressure rate.

---

# Physical Study: Stenosed Channel Flow

To move beyond the idealized lid-driven cavity, the project now includes a more
realistic internal-flow scenario: a 2D channel with a symmetric constriction
representing a mild or severe stenosis.

## Reproducible geometry and tags

The geometry is generated with the Gmsh Python API through
`gmsh.model.occ`. No GUI-only meshing steps are required.

- Base geometry: rectangle of length `4` and height `1`
- Constriction: two semicircular wall intrusions centered at `x = 2`
- Variants:
  - mild stenosis: radius `0.18`
  - severe stenosis: radius `0.30`

Facet tags are assigned reproducibly in the mesh script:

- inlet: tag `1`
- outlet: tag `2`
- walls: tag `3`

The mesh is imported into FEniCSx using:

```python
from dolfinx.io import gmsh
mesh_data = gmsh.read_from_msh(...)
```

## Controlled comparison

Both variants are solved under the same conditions:

- steady incompressible Stokes flow
- Taylor–Hood `P2/P1` discretization
- parabolic inlet profile with mean speed `1.0`
- no-slip walls
- natural outlet with pressure determined up to a small penalty regularization

## Quantities of interest

The primary quantity of interest is:

- pressure drop = average inlet pressure minus average outlet pressure

Secondary quantities are:

- maximum speed on a reproducible sampling grid
- outlet flow rate

## Current results

From [results/physical_study/stenosis_summary.json](/Users/brianhuynh/Project01_MSSD/results/physical_study/stenosis_summary.json:1):

- mild stenosis:
  - pressure drop `1.245225`
  - maximum speed `2.152012`
  - outlet flow rate `1.000000`
- severe stenosis:
  - pressure drop `2.555492`
  - maximum speed `3.566221`
  - outlet flow rate `1.000000`

## Physical interpretation

The severe stenosis performs worse with respect to the pressure-drop metric.
Its narrower throat accelerates the flow more strongly, which raises the local
speed and steepens the pressure gradient through the constriction. The mesh and
field figures show the main physical structures:

- high-speed core through the throat
- strong pressure change across the constriction
- concentrated wall effects near the narrowed region

The mild stenosis performs better because it delivers the same outlet flow with
substantially lower hydraulic resistance.

---

# Benchmark Reference

The cavity centerline comparisons are based on the classical benchmark study:

Ghia, U., Ghia, K. N., and Shin, C. T. (1982).  
*High-Re solutions for incompressible flow using the Navier–Stokes equations
and a multigrid method.*  
Journal of Computational Physics, 48(3), 387–411.
