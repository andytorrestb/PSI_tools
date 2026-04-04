from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class GasSurfaceData:
    r: np.ndarray
    number_density: np.ndarray
    rho: np.ndarray
    phi: np.ndarray
    vn: np.ndarray
    vt: np.ndarray
    velocity: np.ndarray
    normal: np.ndarray
    molecular_mass: float


@dataclass
class DSMCSurfaceExtractorConfig:
    patch_name: str = "floorWall"
    u_field_name: str = "UMean"
    n_field_name: str = "rhoN"
    n_bins: int = 120


class FoamParseError(RuntimeError):
    pass


def list_time_directories(case_dir: Path) -> List[Path]:
    time_dirs: List[Tuple[float, Path]] = []
    for p in case_dir.iterdir():
        if not p.is_dir():
            continue
        try:
            t = float(p.name)
        except ValueError:
            continue
        time_dirs.append((t, p))
    return [p for _, p in sorted(time_dirs, key=lambda x: x[0])]


def latest_time_directory(case_dir: Path) -> Path:
    candidates = list_time_directories(case_dir)
    if not candidates:
        raise FileNotFoundError(f"No numeric time directories found in {case_dir}")
    return candidates[-1]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_block_after_count(text: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not re.fullmatch(r"\s*\d+\s*", line):
            continue

        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            continue

        remainder = "\n".join(lines[j:])
        start = remainder.find("(")
        if start < 0:
            continue

        depth = 0
        end = -1
        for k in range(start, len(remainder)):
            ch = remainder[k]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = k
                    break

        if end > start:
            return remainder[start + 1 : end]

    raise FoamParseError("Could not locate OpenFOAM list block.")


def read_label_list(path: Path) -> np.ndarray:
    txt = _read_text(path)
    block = _extract_block_after_count(txt)
    vals = [int(v) for v in re.findall(r"[-+]?\d+", block)]
    return np.asarray(vals, dtype=np.int64)


def read_face_list(path: Path) -> List[np.ndarray]:
    txt = _read_text(path)
    block = _extract_block_after_count(txt)
    faces: List[np.ndarray] = []
    for size_str, body in re.findall(r"(\d+)\(([^\)]*)\)", block):
        idx = np.fromstring(body, sep=" ", dtype=np.int64)
        expected = int(size_str)
        if idx.size != expected:
            raise FoamParseError(f"Face size mismatch in {path}: expected {expected}, got {idx.size}")
        faces.append(idx)
    return faces


def read_points(path: Path) -> np.ndarray:
    txt = _read_text(path)
    block = _extract_block_after_count(txt)
    pts = []
    for x, y, z in re.findall(r"\(([-+0-9eE\.]+)\s+([-+0-9eE\.]+)\s+([-+0-9eE\.]+)\)", block):
        pts.append((float(x), float(y), float(z)))
    if not pts:
        raise FoamParseError(f"No points parsed from {path}")
    return np.asarray(pts, dtype=float)


def parse_boundary_file(path: Path) -> Dict[str, Dict[str, int | str]]:
    txt = _read_text(path)
    content = txt.split("(", 1)[1].rsplit(")", 1)[0]
    patches: Dict[str, Dict[str, int | str]] = {}
    # Simple parser for OpenFOAM boundary dictionary entries.
    for m in re.finditer(r"(\w+)\s*\{(.*?)\}", content, flags=re.S):
        name = m.group(1)
        body = m.group(2)
        n_faces = re.search(r"nFaces\s+(\d+)\s*;", body)
        start_face = re.search(r"startFace\s+(\d+)\s*;", body)
        p_type = re.search(r"type\s+(\w+)\s*;", body)
        if n_faces and start_face:
            patches[name] = {
                "nFaces": int(n_faces.group(1)),
                "startFace": int(start_face.group(1)),
                "type": p_type.group(1) if p_type else "unknown",
            }
    return patches


def parse_internal_scalar(field_path: Path) -> np.ndarray:
    txt = _read_text(field_path)
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s+\d+\s*\((.*?)\)\s*;", txt, flags=re.S)
    if not m:
        raise FoamParseError(f"Could not parse internal scalar field from {field_path}")
    return np.fromstring(m.group(1), sep=" ", dtype=float)


def parse_internal_vector(field_path: Path) -> np.ndarray:
    txt = _read_text(field_path)
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s+\d+\s*\((.*?)\)\s*;", txt, flags=re.S)
    if not m:
        raise FoamParseError(f"Could not parse internal vector field from {field_path}")
    vecs = re.findall(r"\(([-+0-9eE\.]+)\s+([-+0-9eE\.]+)\s+([-+0-9eE\.]+)\)", m.group(1))
    return np.asarray(vecs, dtype=float)


def parse_patch_vector(field_path: Path, patch_name: str) -> Optional[np.ndarray]:
    txt = _read_text(field_path)
    m = re.search(rf"\b{re.escape(patch_name)}\b\s*\{{(.*?)\}}", txt, flags=re.S)
    if not m:
        return None
    body = m.group(1)
    m_val = re.search(r"value\s+nonuniform\s+List<vector>\s+\d+\s*\((.*?)\)\s*;", body, flags=re.S)
    if not m_val:
        return None
    vecs = re.findall(r"\(([-+0-9eE\.]+)\s+([-+0-9eE\.]+)\s+([-+0-9eE\.]+)\)", m_val.group(1))
    if not vecs:
        return None
    return np.asarray(vecs, dtype=float)


def read_species_mass(case_dir: Path, species_name: Optional[str] = None) -> float:
    txt = _read_text(case_dir / "constant" / "dsmcProperties")
    if species_name is None:
        m_species = re.search(r"typeIdList\s*\((\w+)\)", txt)
        if not m_species:
            raise FoamParseError("Could not infer species name from dsmcProperties")
        species_name = m_species.group(1)

    species_block = re.search(rf"\b{re.escape(species_name)}\b\s*\{{(.*?)\}}", txt, flags=re.S)
    if not species_block:
        raise FoamParseError(f"Species {species_name} not found in dsmcProperties")

    mass_match = re.search(r"mass\s+([-+0-9eE\.]+)\s*;", species_block.group(1))
    if not mass_match:
        raise FoamParseError(f"mass entry for species {species_name} not found")
    return float(mass_match.group(1))


def _face_normal(points: np.ndarray, face: np.ndarray) -> np.ndarray:
    if face.size < 3:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    p0 = points[face[0]]
    p1 = points[face[1]]
    p2 = points[face[2]]
    n = np.cross(p1 - p0, p2 - p0)
    mag = np.linalg.norm(n)
    if mag <= 1e-15:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return n / mag


def _safe_bin_average(r: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.digitize(r, edges) - 1
    idx = np.clip(idx, 0, edges.size - 2)
    out = np.zeros(edges.size - 1, dtype=float)
    counts = np.zeros(edges.size - 1, dtype=float)
    for i in range(values.size):
        b = idx[i]
        out[b] += values[i]
        counts[b] += 1.0
    valid = counts > 0
    out[valid] /= counts[valid]
    if np.any(~valid):
        centers = 0.5 * (edges[:-1] + edges[1:])
        out[~valid] = np.interp(centers[~valid], centers[valid], out[valid], left=out[valid][0], right=out[valid][-1])
    return out


def _safe_bin_average_vector(r: np.ndarray, vectors: np.ndarray, edges: np.ndarray) -> np.ndarray:
    cols = [_safe_bin_average(r, vectors[:, j], edges) for j in range(vectors.shape[1])]
    return np.column_stack(cols)


def _project_velocity_components(velocity: np.ndarray, normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v_dot_n = np.einsum("ij,ij->i", velocity, normal)
    vn = np.abs(v_dot_n)
    v_t_vec = velocity - v_dot_n[:, None] * normal
    vt = np.linalg.norm(v_t_vec, axis=1)
    return vn, vt


def load_surface_data_from_case(
    case_dir: Path,
    time_dir: Optional[Path] = None,
    cfg: Optional[DSMCSurfaceExtractorConfig] = None,
) -> GasSurfaceData:
    cfg = cfg or DSMCSurfaceExtractorConfig()
    case_dir = Path(case_dir)
    time_dir = time_dir or latest_time_directory(case_dir)

    mesh_dir = case_dir / "constant" / "polyMesh"
    boundary_info = parse_boundary_file(mesh_dir / "boundary")
    if cfg.patch_name not in boundary_info:
        raise FoamParseError(f"Patch {cfg.patch_name} not found in polyMesh/boundary")

    patch = boundary_info[cfg.patch_name]
    start_face = int(patch["startFace"])
    n_faces = int(patch["nFaces"])
    end_face = start_face + n_faces

    owner = read_label_list(mesh_dir / "owner")
    faces = read_face_list(mesh_dir / "faces")
    points = read_points(mesh_dir / "points")

    rho_n_internal = parse_internal_scalar(time_dir / cfg.n_field_name)
    u_internal = parse_internal_vector(time_dir / cfg.u_field_name)
    u_patch = parse_patch_vector(time_dir / cfg.u_field_name, cfg.patch_name)

    patch_face_indices = np.arange(start_face, end_face, dtype=np.int64)
    owner_cells = owner[patch_face_indices]

    face_centers = np.zeros((n_faces, 3), dtype=float)
    face_normals = np.zeros((n_faces, 3), dtype=float)
    for i, f_idx in enumerate(patch_face_indices):
        f = faces[int(f_idx)]
        verts = points[f]
        face_centers[i] = np.mean(verts, axis=0)
        face_normals[i] = _face_normal(points, f)

    r = np.sqrt(face_centers[:, 0] ** 2 + face_centers[:, 1] ** 2)
    number_density = rho_n_internal[owner_cells]
    velocity = u_patch if (u_patch is not None and u_patch.shape[0] == n_faces) else u_internal[owner_cells]

    molecular_mass = read_species_mass(case_dir)
    rho = number_density * molecular_mass
    vn, vt = _project_velocity_components(velocity, face_normals)
    phi = number_density * vn

    if cfg.n_bins > 0 and n_faces > cfg.n_bins:
        r_min = float(np.min(r))
        r_max = float(np.max(r))
        edges = np.linspace(r_min, r_max + 1e-12, cfg.n_bins + 1)
        r_b = 0.5 * (edges[:-1] + edges[1:])
        number_density_b = _safe_bin_average(r, number_density, edges)
        rho_b = _safe_bin_average(r, rho, edges)
        vn_b = _safe_bin_average(r, vn, edges)
        vt_b = _safe_bin_average(r, vt, edges)
        phi_b = _safe_bin_average(r, phi, edges)
        vel_b = _safe_bin_average_vector(r, velocity, edges)
        nrm_b = _safe_bin_average_vector(r, face_normals, edges)
        nrm_mag = np.linalg.norm(nrm_b, axis=1)
        nrm_mag[nrm_mag < 1e-16] = 1.0
        nrm_b = nrm_b / nrm_mag[:, None]
        return GasSurfaceData(
            r=r_b,
            number_density=number_density_b,
            rho=rho_b,
            phi=phi_b,
            vn=vn_b,
            vt=vt_b,
            velocity=vel_b,
            normal=nrm_b,
            molecular_mass=molecular_mass,
        )

    return GasSurfaceData(
        r=r,
        number_density=number_density,
        rho=rho,
        phi=phi,
        vn=vn,
        vt=vt,
        velocity=velocity,
        normal=face_normals,
        molecular_mass=molecular_mass,
    )


def synthetic_surface_profile(
    n_points: int = 150,
    r_max: float = 20.0,
    molecular_mass: float = 66.3e-27,
    n0: float = 2.0e20,
    v0: float = 1800.0,
    plume_angle_deg: float = 30.0,
) -> GasSurfaceData:
    r = np.linspace(0.0, r_max, n_points)
    number_density = n0 * np.exp(-r / (0.18 * r_max + 1e-12))
    speed = v0 * np.exp(-r / (0.35 * r_max + 1e-12))
    angle = np.deg2rad(plume_angle_deg)
    vn = np.maximum(speed * np.cos(angle), 0.0)
    vt = np.maximum(speed * np.sin(angle), 0.0)
    phi = number_density * vn
    rho = number_density * molecular_mass

    velocity = np.zeros((n_points, 3), dtype=float)
    velocity[:, 0] = vt
    velocity[:, 2] = -vn
    normal = np.zeros((n_points, 3), dtype=float)
    normal[:, 2] = 1.0

    return GasSurfaceData(
        r=r,
        number_density=number_density,
        rho=rho,
        phi=phi,
        vn=vn,
        vt=vt,
        velocity=velocity,
        normal=normal,
        molecular_mass=molecular_mass,
    )
