from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


FLUID_TAG = 1
INLET_TAG = 1
OUTLET_TAG = 2
WALL_TAG = 3


@dataclass(frozen=True)
class ChannelVariant:
    name: str
    notch_radius: float
    mesh_size: float


CHANNEL_LENGTH = 4.0
CHANNEL_HEIGHT = 1.0
STENOSIS_CENTER_X = 2.0

VARIANTS = {
    "mild": ChannelVariant("mild", notch_radius=0.18, mesh_size=0.07),
    "severe": ChannelVariant("severe", notch_radius=0.30, mesh_size=0.06),
}


def build_constricted_channel_mesh(mesh_file: Path, variant: ChannelVariant) -> Path:
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"constricted_channel_{variant.name}")

    rectangle = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, CHANNEL_LENGTH, CHANNEL_HEIGHT)
    bottom_notch = gmsh.model.occ.addDisk(
        STENOSIS_CENTER_X, 0.0, 0.0, variant.notch_radius, variant.notch_radius
    )
    top_notch = gmsh.model.occ.addDisk(
        STENOSIS_CENTER_X, CHANNEL_HEIGHT, 0.0, variant.notch_radius, variant.notch_radius
    )
    cut_result, _ = gmsh.model.occ.cut(
        [(2, rectangle)],
        [(2, bottom_notch), (2, top_notch)],
        removeObject=True,
        removeTool=True,
    )
    gmsh.model.occ.synchronize()

    if len(cut_result) != 1:
        raise RuntimeError(f"Expected one fluid surface, got {cut_result!r}")
    fluid_surface = cut_result[0][1]

    gmsh.model.addPhysicalGroup(2, [fluid_surface], FLUID_TAG)
    gmsh.model.setPhysicalName(2, FLUID_TAG, "fluid")

    inlet_curves: list[int] = []
    outlet_curves: list[int] = []
    wall_curves: list[int] = []
    boundary_entities = gmsh.model.getBoundary([(2, fluid_surface)], oriented=False)
    for dim, entity in boundary_entities:
        if dim != 1:
            continue
        center_x, center_y, _ = gmsh.model.occ.getCenterOfMass(dim, entity)
        if abs(center_x - 0.0) < 1.0e-8:
            inlet_curves.append(entity)
        elif abs(center_x - CHANNEL_LENGTH) < 1.0e-8:
            outlet_curves.append(entity)
        else:
            wall_curves.append(entity)

    gmsh.model.addPhysicalGroup(1, inlet_curves, INLET_TAG)
    gmsh.model.setPhysicalName(1, INLET_TAG, "inlet")
    gmsh.model.addPhysicalGroup(1, outlet_curves, OUTLET_TAG)
    gmsh.model.setPhysicalName(1, OUTLET_TAG, "outlet")
    gmsh.model.addPhysicalGroup(1, wall_curves, WALL_TAG)
    gmsh.model.setPhysicalName(1, WALL_TAG, "walls")

    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.4 * variant.mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", variant.mesh_size)

    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", wall_curves)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", 0.4 * variant.mesh_size)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", variant.mesh_size)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.08)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 0.4)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

    gmsh.model.mesh.generate(2)
    mesh_file.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(mesh_file))
    gmsh.finalize()
    return mesh_file


def generate_all_meshes(mesh_dir: Path) -> dict[str, Path]:
    return {
        name: build_constricted_channel_mesh(mesh_dir / f"constricted_channel_{name}.msh", variant)
        for name, variant in VARIANTS.items()
    }


def main():
    project_root = Path(__file__).resolve().parent
    mesh_dir = project_root / "results" / "physical_study" / "meshes"
    mesh_paths = generate_all_meshes(mesh_dir)
    for name, mesh_path in mesh_paths.items():
        print(f"{name}: {mesh_path}")


if __name__ == "__main__":
    main()
