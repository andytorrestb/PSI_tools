"""Particle trajectory solver replicating Fontes et al. (2022), Section 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml
from fluidfoam import readfield, readmesh


ArrayLike = np.ndarray


@dataclass(frozen=True)
class Domain:
    """Axis-aligned domain limits used for escape detection."""

    x_max: float
    y_max: float
    z_min: float
    z_max: float


def find_nearest_cell(x_p: np.ndarray, cell_centres: np.ndarray) -> int:
    """Return the nearest cell centroid index (Fontes et al. 2022, Section 3).

    This reproduces the brute-force tracker used in the paper.
    """
    # O(N_cells) brute-force sweep per sub-step; use scipy.spatial.KDTree in production.
    deltas = cell_centres - x_p
    dist2 = np.einsum("ij,ij->i", deltas, deltas)
    return int(np.argmin(dist2))


def compute_reynolds(
    rho: float,
    rel_speed: float,
    d_p: float,
    temperature: float,
    drag_A: float,
    drag_beta: float,
) -> float:
    """Compute particle Reynolds number (Eq. after Eq. 6, Fontes et al. 2022)."""
    if temperature <= 0.0 or d_p <= 0.0:
        return 0.0
    return float((rho * rel_speed * d_p) / (drag_A * (temperature**drag_beta)))


def compute_cd(re_p: float) -> float:
    """Compute drag coefficient using Lane et al. correlation (Eq. 6)."""
    if re_p <= 0.0:
        return 0.0
    if re_p < 2.0:
        return float(24.0 * re_p ** (-1.0))
    if re_p < 500.0:
        return float(18.5 * re_p ** (-0.6))
    return 0.44


def compute_drag(
    u_g: np.ndarray,
    u_p: np.ndarray,
    rho: float,
    rho_p: float,
    d_p: float,
    temperature: float,
    drag_A: float,
    drag_beta: float,
) -> np.ndarray:
    """Compute drag acceleration contribution from Eq. 5 in Fontes et al. (2022).

    Eq. 5 is force per particle mass, so this function returns acceleration [m/s^2].
    """
    rel = u_g - u_p
    rel_speed = float(np.linalg.norm(rel))
    re_p = compute_reynolds(rho, rel_speed, d_p, temperature, drag_A, drag_beta)
    c_d = compute_cd(re_p)
    factor = (3.0 * rho * c_d) / (4.0 * rho_p * d_p) if d_p > 0.0 else 0.0
    return factor * rel_speed * rel


def compute_weight(rho: float, rho_p: float, gravity_m_s2: float) -> np.ndarray:
    """Compute buoyancy-weight acceleration from Eq. 7 in Fontes et al. (2022)."""
    g_vec = np.array([0.0, 0.0, -abs(gravity_m_s2)], dtype=float)
    return (1.0 - (rho / rho_p)) * g_vec


def in_domain(position: np.ndarray, domain: Domain) -> bool:
    """Check if a particle position is inside the configured tracking domain."""
    x, y, z = position
    return (
        (-domain.x_max <= x <= domain.x_max)
        and (-domain.y_max <= y <= domain.y_max)
        and (domain.z_min <= z <= domain.z_max)
    )


def rk4_step(
    state: np.ndarray,
    dt: float,
    cell_centres: np.ndarray,
    u_field: np.ndarray,
    rho_field: np.ndarray,
    t_field: np.ndarray,
    domain: Domain,
    d_p: float,
    rho_p: float,
    gravity_m_s2: float,
    drag_A: float,
    drag_beta: float,
) -> Tuple[np.ndarray, bool]:
    """Advance one RK4 step for state [x,y,z,ux,uy,uz] (Section 3, Fontes et al. 2022)."""

    def deriv(y: np.ndarray) -> Tuple[np.ndarray, bool]:
        pos = y[:3]
        vel = y[3:]

        # Per prompt: bounding-box check before any nearest-cell call.
        if not in_domain(pos, domain):
            return np.zeros(6, dtype=float), True

        idx = find_nearest_cell(pos, cell_centres)
        u_g = u_field[idx]
        rho = float(rho_field[idx])
        temperature = float(t_field[idx])

        a_drag = compute_drag(u_g, vel, rho, rho_p, d_p, temperature, drag_A, drag_beta)
        a_weight = compute_weight(rho, rho_p, gravity_m_s2)

        out = np.zeros(6, dtype=float)
        out[:3] = vel
        out[3:] = a_drag + a_weight
        return out, False

    k1, esc1 = deriv(state)
    if esc1:
        return state.copy(), True

    k2_state = state + 0.5 * dt * k1
    k2, esc2 = deriv(k2_state)
    if esc2:
        return k2_state.copy(), True

    k3_state = state + 0.5 * dt * k2
    k3, esc3 = deriv(k3_state)
    if esc3:
        return k3_state.copy(), True

    k4_state = state + dt * k3
    k4, esc4 = deriv(k4_state)
    if esc4:
        return k4_state.copy(), True

    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return next_state, False


def load_openfoam_fields(case_dir: str, time_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load mesh centres and fields U, rho, T from an OpenFOAM case via fluidfoam.

    OpenFOAM static meshes live in constant/polyMesh; some fluidfoam paths look in
    <time>/polyMesh when time_name is provided. We therefore read mesh from constant
    first and only fall back to time/polyMesh for moving-mesh cases.
    """
    try:
        x_c, y_c, z_c = readmesh(case_dir)
    except FileNotFoundError:
        x_c, y_c, z_c = readmesh(case_dir, time_name=time_dir)
    cell_centres = np.column_stack((np.ravel(x_c), np.ravel(y_c), np.ravel(z_c))).astype(float)

    u_raw = np.asarray(readfield(case_dir, time_name=time_dir, name="UMean"), dtype=float)
    if u_raw.ndim != 2:
        raise ValueError("Field U must be a rank-2 array from fluidfoam.readfield().")
    if u_raw.shape[0] == 3:
        u_field = u_raw.T.copy()
    elif u_raw.shape[1] == 3:
        u_field = u_raw.copy()
    else:
        raise ValueError("Field U must have one dimension of size 3.")

    rho_field = np.ravel(np.asarray(readfield(case_dir, time_name=time_dir, name="rhoM"), dtype=float))
    t_field = np.ravel(np.asarray(readfield(case_dir, time_name=time_dir, name="TranslationalT"), dtype=float))

    n_cells = cell_centres.shape[0]
    # Some DSMC exports carry one extra trailing entry in cell fields.
    if u_field.shape[0] == n_cells + 1 and rho_field.shape[0] == n_cells + 1 and t_field.shape[0] == n_cells + 1:
        u_field = u_field[:n_cells, :]
        rho_field = rho_field[:n_cells]
        t_field = t_field[:n_cells]

    if not (u_field.shape[0] == n_cells == rho_field.shape[0] == t_field.shape[0]):
        raise ValueError(
            "Mesh and field size mismatch: "
            f"centres={n_cells}, UMean={u_field.shape[0]}, rhoM={rho_field.shape[0]}, TranslationalT={t_field.shape[0]}"
        )

    return cell_centres, u_field, rho_field, t_field


def make_case_key(d_p: float, r_p0: float) -> str:
    """Return a stable string key for np.savez serialization."""
    return f"dp={d_p:.8e}|r0={r_p0:.8e}"


def simulate_particle(
    d_p: float,
    r0: float,
    cfg: Dict[str, Any],
    cell_centres: np.ndarray,
    u_field: np.ndarray,
    rho_field: np.ndarray,
    t_field: np.ndarray,
) -> Dict[str, Any]:
    """Integrate one particle trajectory until escaped, grounded, or max time."""
    domain = Domain(
        x_max=float(cfg["domain"]["x_max"]),
        y_max=float(cfg["domain"]["y_max"]),
        z_min=float(cfg["domain"]["z_min"]),
        z_max=float(cfg["domain"]["z_max"]),
    )

    dt = float(cfg["integration"]["dt_s"])
    t_max = float(cfg["integration"]["t_max_s"])
    gravity = float(cfg["physics"]["gravity_m_s2"])
    drag_A = float(cfg["physics"]["drag_A"])
    drag_beta = float(cfg["physics"]["drag_beta"])
    rho_p = float(cfg["particle"]["density_kg_m3"])

    state = np.array([r0, 0.0, float(cfg["particle"]["initial_height_m"]), 0.0, 0.0, 0.0], dtype=float)
    trajectory = [state.copy()]

    t = 0.0
    status = "max_time"
    grounded_collisions = 0
    ground_vtol = 0.05
    grounded_collisions_needed = 5

    while t < t_max:
        next_state, escaped = rk4_step(
            state,
            dt,
            cell_centres,
            u_field,
            rho_field,
            t_field,
            domain,
            d_p,
            rho_p,
            gravity,
            drag_A,
            drag_beta,
        )

        if escaped:
            status = "escaped"
            state = next_state
            trajectory.append(state.copy())
            break

        if next_state[2] <= domain.z_min:
            next_state[2] = domain.z_min
            incoming_uz = next_state[5]
            next_state[5] = -incoming_uz
            if abs(incoming_uz) <= ground_vtol:
                grounded_collisions += 1
            else:
                grounded_collisions = 0

            if grounded_collisions >= grounded_collisions_needed:
                status = "grounded"
                state = next_state
                trajectory.append(state.copy())
                break

        state = next_state
        t += dt
        trajectory.append(state.copy())

    trajectory_arr = np.asarray(trajectory, dtype=float)
    return {
        "trajectory": trajectory_arr,
        "exit_position": trajectory_arr[-1, :3].copy(),
        "exit_velocity": trajectory_arr[-1, 3:].copy(),
        "exit_time": float(t),
        "status": status,
    }


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_batch(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Run full (diameter, radial-position) sweep and return serializable results."""
    case_dir = str(cfg["openfoam"]["case_dir"])
    time_dir = str(cfg["openfoam"]["time_dir"])

    cell_centres, u_field, rho_field, t_field = load_openfoam_fields(case_dir, time_dir)

    diameters = [float(v) for v in cfg["particle"]["diameters_m"]]
    radial_positions = [float(v) for v in cfg["particle"]["radial_positions_m"]]

    results: Dict[str, Dict[str, Any]] = {}

    for d_p in diameters:
        for r0 in radial_positions:
            case_result = simulate_particle(d_p, r0, cfg, cell_centres, u_field, rho_field, t_field)
            speed = float(np.linalg.norm(case_result["exit_velocity"]))
            print(
                f"Completed dp={d_p:.2e} m | r0={r0:.1f} m | "
                f"status={case_result['status']} | |v|={speed:.1f} m/s"
            )

            results[make_case_key(d_p, r0)] = case_result

    return results


def save_results(results: Dict[str, Dict[str, Any]], results_file: str) -> None:
    """Save dictionary to compressed NumPy archive as object entries."""
    to_save: Dict[str, np.ndarray] = {}
    for key, value in results.items():
        to_save[key] = np.array(value, dtype=object)

    out_path = Path(results_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **to_save)


def main(config_path: str = "config.yaml") -> None:
    """CLI entry point for batch execution."""
    cfg = load_config(config_path)
    results = run_batch(cfg)
    save_results(results, str(cfg["output"]["results_file"]))


if __name__ == "__main__":
    main()
