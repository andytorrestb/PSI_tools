"""
Builds a quarter-cylinder blockMeshDict using classy_blocks.

The script writes system/blockMeshDict for the local case by default.
Adjust parameters via a YAML config file. No commands are executed amatically.
"""

import os
import argparse
import yaml

import classy_blocks as cb
from classy_blocks.util import functions as f


def build_quarter_cylinder(
    length: float,
    radius: float,
    axial_cells: int,
    radial_cells: int,
    tangential_cells: int,
    wall_thickness: float | None,
    wall_patch: str,
    symmetry_patch: str,
    symmetry_patch_type: str,
    start_patch: str,
    end_patch: str,
    output_path: str,
    debug_vtk: str | None,
) -> str:
    """Constructs the mesh and writes blockMeshDict; returns its path."""
    mesh = cb.Mesh()

    # Axis along +z; quarter spans +x/+y. Radius point must be perpendicular to axis.
    axis_point_1 = f.vector(0.0, 0.0, 0.0)
    axis_point_2 = f.vector(0.0, 0.0, length)
    radius_point_1 = f.vector(radius, 0.0, 0.0)

    quarter_cyl = cb.QuarterCylinder(axis_point_1, axis_point_2, radius_point_1)

    quarter_cyl.chop_axial(count=axial_cells)

    if wall_thickness is None:
        quarter_cyl.chop_radial(count=radial_cells)
    else:
        # end_size near the wall gives a thin first cell; expansion keeps grading mild.
        quarter_cyl.chop_radial(count=radial_cells, end_size=wall_thickness, c2c_expansion=1.2)

    quarter_cyl.chop_tangential(count=tangential_cells)

    quarter_cyl.set_start_patch(start_patch)
    quarter_cyl.set_end_patch(end_patch)
    quarter_cyl.set_outer_patch(wall_patch)
    quarter_cyl.set_symmetry_patch(symmetry_patch)

    mesh.add(quarter_cyl)
    mesh.set_default_patch("default", wall_patch)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh.write(output_path, debug_path=debug_vtk)

    # classy_blocks writes symmetry patches with type "patch"; we override the boundary
    # type to the desired symmetry-type (e.g., "symmetry") after writing.
    enforce_patch_type(output_path, symmetry_patch, symmetry_patch_type)
    return output_path


def enforce_patch_type(blockmesh_path: str, patch_name: str, patch_type: str) -> None:
    """Ensure the named patch uses the desired type inside blockMeshDict."""
    with open(blockmesh_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_patch = False
    seen_name = False
    changed = False
    updated: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not in_patch:
            if stripped == patch_name:
                seen_name = True
            elif seen_name and stripped.startswith("{"):
                in_patch = True
                seen_name = False

        if in_patch and stripped.startswith("type"):
            indent = line[: line.find("type")]
            line = f"{indent}type {patch_type};\n"
            changed = True
        if in_patch and stripped.startswith("}"):
            in_patch = False
        updated.append(line)

    with open(blockmesh_path, "w", encoding="utf-8") as f:
        f.writelines(updated)

    if not changed:
        raise RuntimeError(
            f"Failed to set patch '{patch_name}' type to '{patch_type}' in {blockmesh_path}."
        )


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quarter-cylinder mesh generator using classy_blocks")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: config.yaml next to this script)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config or os.path.join(script_dir, "config.yaml")

    cfg = load_config(config_path)

    output = cfg.get("output")
    if output is None:
        output = os.path.abspath(os.path.join(script_dir, "..", "system", "blockMeshDict"))

    # Force symmetry patch type to "symmetry" unless explicitly overridden differently.
    sym_type = cfg.get("symmetry_patch_type") or "symmetry"
    if str(sym_type).lower() == "symmetryplane":
        sym_type = "symmetry"

    blockmesh_path = build_quarter_cylinder(
        length=cfg.get("length", 1.0),
        radius=cfg.get("radius", 0.5),
        axial_cells=cfg.get("axial_cells", 20),
        radial_cells=cfg.get("radial_cells", 8),
        tangential_cells=cfg.get("tangential_cells", 12),
        wall_thickness=cfg.get("wall_thickness"),
        wall_patch=cfg.get("wall_patch", "solidCylinder"),
        symmetry_patch=cfg.get("symmetry_patch", "symmetryPlane"),
        symmetry_patch_type=sym_type,
        start_patch=cfg.get("start_patch", "inlet"),
        end_patch=cfg.get("end_patch", "topOutlet"),
        output_path=output,
        debug_vtk=cfg.get("debug_vtk"),
    )

    print(f"blockMeshDict written to: {blockmesh_path}")
    if cfg.get("debug_vtk"):
        print(f"Debug VTK written to: {cfg['debug_vtk']}")


if __name__ == "__main__":
    main()
