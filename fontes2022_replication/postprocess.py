"""Post-process Fontes-style particle trajectory batch outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml


def parse_case_key(key: str) -> Tuple[float, float]:
    """Parse serialized result key format: dp=<float>|r0=<float>."""
    parts = key.split("|")
    d_p = float(parts[0].split("=")[1])
    r0 = float(parts[1].split("=")[1])
    return d_p, r0


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_results(path: str) -> Dict[Tuple[float, float], Dict[str, Any]]:
    """Load .npz results and return a dictionary keyed by (d_p, r0)."""
    raw = np.load(path, allow_pickle=True)
    out: Dict[Tuple[float, float], Dict[str, Any]] = {}
    for key in raw.files:
        d_p, r0 = parse_case_key(key)
        out[(d_p, r0)] = raw[key].item()
    return out


def particle_mass(d_p: float, rho_p: float) -> float:
    """Compute spherical particle mass from diameter and material density."""
    radius = 0.5 * d_p
    return (4.0 / 3.0) * np.pi * (radius**3) * rho_p


def build_metrics(
    results: Dict[Tuple[float, float], Dict[str, Any]],
    diameters: List[float],
    radial_positions: List[float],
    rho_p: float,
) -> Dict[float, Dict[str, np.ndarray]]:
    """Build arrays of velocity magnitude, angle, and kinetic energy by r0 line."""
    metrics: Dict[float, Dict[str, np.ndarray]] = {}

    for r0 in radial_positions:
        speeds = []
        angles_deg = []
        energies = []

        for d_p in diameters:
            entry = results[(d_p, r0)]
            v = np.asarray(entry["exit_velocity"], dtype=float)
            speed = float(np.linalg.norm(v))
            angle_deg = float(np.degrees(np.arctan2(v[2], v[0])))
            ke = 0.5 * particle_mass(d_p, rho_p) * speed * speed

            speeds.append(speed)
            angles_deg.append(angle_deg)
            energies.append(ke)

        metrics[r0] = {
            "speed": np.asarray(speeds, dtype=float),
            "angle_deg": np.asarray(angles_deg, dtype=float),
            "ke": np.asarray(energies, dtype=float),
        }

    return metrics


def make_plots(cfg: Dict[str, Any], results: Dict[Tuple[float, float], Dict[str, Any]]) -> None:
    """Generate three replication plots and save to configured directory."""
    diameters = [float(v) for v in cfg["particle"]["diameters_m"]]
    radial_positions = [float(v) for v in cfg["particle"]["radial_positions_m"]]
    rho_p = float(cfg["particle"]["density_kg_m3"])

    plots_dir = Path(str(cfg["output"]["plots_dir"]))
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = build_metrics(results, diameters, radial_positions, rho_p)

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(radial_positions) - 1, 1)) for i in range(len(radial_positions))]

    plt.figure(figsize=(9, 6))
    for idx, r0 in enumerate(radial_positions):
        plt.plot(diameters, metrics[r0]["speed"], color=colors[idx], label=f"r0={r0:.1f} m")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Particle diameter, d_p [m]")
    plt.ylabel("Exit speed, |v_p| [m/s]")
    plt.title("Velocity Magnitude vs Particle Diameter")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "velocity_vs_diameter.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    for idx, r0 in enumerate(radial_positions):
        plt.plot(diameters, metrics[r0]["angle_deg"], color=colors[idx], label=f"r0={r0:.1f} m")
    plt.xscale("log")
    plt.xlabel("Particle diameter, d_p [m]")
    plt.ylabel("Ejection angle [deg]")
    plt.title("Ejection Angle vs Particle Diameter")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "angle_vs_diameter.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    for idx, r0 in enumerate(radial_positions):
        plt.plot(diameters, metrics[r0]["ke"], color=colors[idx], label=f"r0={r0:.1f} m")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Particle diameter, d_p [m]")
    plt.ylabel("Kinetic energy [J]")
    plt.title("Kinetic Energy vs Particle Diameter")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "kinetic_energy_vs_diameter.png", dpi=180)
    plt.close()

    traj_dir = plots_dir / "trajectories_by_diameter"
    traj_dir.mkdir(parents=True, exist_ok=True)

    for d_p in diameters:
        plt.figure(figsize=(9, 6))
        for idx, r0 in enumerate(radial_positions):
            trajectory = np.asarray(results[(d_p, r0)]["trajectory"], dtype=float)
            r_track = np.sqrt(trajectory[:, 0] ** 2 + trajectory[:, 1] ** 2)
            z_track = trajectory[:, 2]
            plt.plot(r_track, z_track, color=colors[idx], label=f"r0={r0:.1f} m")

        plt.xlabel("Radial distance, r [m]")
        plt.ylabel("Height, z [m]")
        plt.title(f"Trajectory Map at d_p={d_p:.2e} m")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()

        dp_label = f"{d_p:.2e}".replace("+", "")
        plt.savefig(traj_dir / f"trajectory_dp_{dp_label}.png", dpi=180)
        plt.close()


def main(config_path: str = "config.yaml") -> None:
    """CLI entry point for generating plots from saved solver outputs."""
    cfg = load_config(config_path)
    results = load_results(str(cfg["output"]["results_file"]))
    make_plots(cfg, results)


if __name__ == "__main__":
    main()
