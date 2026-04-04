from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from gas_model import DSMCSurfaceExtractorConfig, FoamParseError, load_surface_data_from_case, synthetic_surface_profile
from psi_model import run_psi_model, sweep_particle_size, sweep_plume_angle, sweep_velocity_scale
from regolith_model import RegolithProperties, default_cohesion_model


def _integrate_profile(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


@dataclass
class SimulationConfig:
    case_dir: Path
    use_dsmc_case: bool = True
    patch_name: str = "floorWall"
    n_bins: int = 140
    particle_diameter: float = 120e-6
    grain_density: float = 3100.0
    gravity: float = 1.62
    c_e: float = 2.0e-24
    plume_angle_deg: float = 35.0
    velocity_scale: float = 1.0
    show_plots: bool = True


def parse_args() -> argparse.Namespace:
    default_case = Path(__file__).resolve().parent.parent / "FontesTest"
    parser = argparse.ArgumentParser(description="DSMC-based PSI prototype for gas-regolith coupling")
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=default_case,
        help="Path to an OpenFOAM DSMC case directory (default: ../FontesTest)",
    )
    parser.add_argument(
        "--particle-diameter",
        type=float,
        default=120e-6,
        help="Grain diameter [m]",
    )
    parser.add_argument(
        "--velocity-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to extracted velocity profile",
    )
    parser.add_argument(
        "--plume-angle-deg",
        type=float,
        default=35.0,
        help="Override plume angle for velocity decomposition [deg]",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable matplotlib plotting",
    )
    return parser.parse_args()


def load_gas_data(cfg: SimulationConfig):
    if cfg.use_dsmc_case:
        try:
            gas = load_surface_data_from_case(
                case_dir=cfg.case_dir,
                cfg=DSMCSurfaceExtractorConfig(
                    patch_name=cfg.patch_name,
                    n_bins=cfg.n_bins,
                ),
            )
            return gas, "dsmc"
        except (FileNotFoundError, FoamParseError) as exc:
            print(f"[WARN] DSMC extraction failed: {exc}")
            print("[WARN] Falling back to synthetic profile for validation.")

    gas = synthetic_surface_profile(
        n_points=max(cfg.n_bins, 100),
        plume_angle_deg=cfg.plume_angle_deg,
    )
    return gas, "synthetic"


def apply_velocity_and_angle(gas, velocity_scale: float, plume_angle_deg: Optional[float] = None):
    if plume_angle_deg is None:
        vn = np.maximum(gas.vn * velocity_scale, 0.0)
        vt = np.maximum(gas.vt * velocity_scale, 0.0)
    else:
        speed = np.sqrt(gas.vn ** 2 + gas.vt ** 2) * velocity_scale
        ang = np.deg2rad(plume_angle_deg)
        vn = np.maximum(speed * np.cos(ang), 0.0)
        vt = np.maximum(speed * np.sin(ang), 0.0)

    gas.vn = vn
    gas.vt = vt
    gas.phi = gas.number_density * vn
    return gas


def sanity_checks(gas, regolith: RegolithProperties, d_p: float, c_e: float) -> None:
    gas_zero = synthetic_surface_profile(n_points=80, v0=0.0)
    zero_res = run_psi_model(gas_zero, regolith, d_p=d_p, c_e=c_e)
    assert np.allclose(zero_res.erosion_rate, 0.0), "Zero velocity should produce zero erosion"

    # Isolate Vt sensitivity by holding vn and number density fixed.
    gas_base = synthetic_surface_profile(n_points=80, v0=900.0, plume_angle_deg=35.0)
    low_res = run_psi_model(gas_base, regolith, d_p=d_p, c_e=c_e)

    gas_high_vt = synthetic_surface_profile(n_points=80, v0=900.0, plume_angle_deg=35.0)
    gas_high_vt.vt = gas_base.vt * 1.5
    gas_high_vt.phi = gas_high_vt.number_density * gas_high_vt.vn
    high_res = run_psi_model(gas_high_vt, regolith, d_p=d_p, c_e=c_e)

    assert _integrate_profile(high_res.erosion_rate, high_res.r) >= _integrate_profile(low_res.erosion_rate, low_res.r), (
        "Increasing Vt with fixed Vn should not reduce erosion in this model"
    )

    # Check size trend in a cohesion-free control, where F_drive ~ d^2 and F_g ~ d^3.
    regolith_no_cohesion = RegolithProperties(
        grain_density=regolith.grain_density,
        gravity=regolith.gravity,
        cohesion_model=None,
        cohesion_constant=0.0,
    )
    small_d = run_psi_model(gas, regolith_no_cohesion, d_p=50e-6, c_e=c_e)
    large_d = run_psi_model(gas, regolith_no_cohesion, d_p=400e-6, c_e=c_e)
    assert np.mean(large_d.safety_factor) <= np.mean(small_d.safety_factor), (
        "Larger grains should be harder to lift in the cohesion-free control case"
    )


def make_plots(base_res, sweep_d, sweep_v, source_tag: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axes[0, 0]
    ax.step(base_res.r, base_res.lift_mask.astype(int), where="mid")
    ax.set_xlabel("Radial distance r [m]")
    ax.set_ylabel("Lift-off (0/1)")
    ax.set_title(f"Lift-off map ({source_tag})")
    ax.set_ylim(-0.05, 1.05)

    ax = axes[0, 1]
    ax.plot(base_res.r, base_res.safety_factor, lw=2)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("Radial distance r [m]")
    ax.set_ylabel("Safety factor")
    ax.set_title("Safety factor distribution")

    ax = axes[1, 0]
    ax.plot(base_res.r, base_res.erosion_rate, lw=2)
    ax.set_xlabel("Radial distance r [m]")
    ax.set_ylabel("Erosion rate")
    ax.set_title("Erosion rate vs radial distance")

    ax = axes[1, 1]
    ax.plot(sweep_d["d_p"] * 1e6, sweep_d["erosion_integral"], label="vs particle size", lw=2)
    vel_scaled = sweep_v["velocity_scale"]
    erosion_v = sweep_v["erosion_integral"]
    # Show the velocity sweep on a twinned x-axis for readability.
    ax2 = ax.twiny()
    ax2.plot(vel_scaled, erosion_v, color="tab:red", lw=2, label="vs velocity")
    ax.set_xlabel("Particle diameter [um]")
    ax2.set_xlabel("Velocity scale [-]")
    ax.set_ylabel("Integrated erosion")
    ax.set_title("Sensitivity: particle size and velocity")

    fig2, ax_ej = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
    ax_ej.plot(base_res.r, base_res.ejection_speed, lw=2)
    ax_ej.set_xlabel("Radial distance r [m]")
    ax_ej.set_ylabel("Ejection speed [m/s]")
    ax_ej.set_title("Optional ejection speed profile")

    plt.show()


def main() -> None:
    args = parse_args()
    cfg = SimulationConfig(
        case_dir=args.case_dir,
        particle_diameter=args.particle_diameter,
        velocity_scale=args.velocity_scale,
        plume_angle_deg=args.plume_angle_deg,
        show_plots=not args.no_plots,
    )

    gas, source_tag = load_gas_data(cfg)
    gas = apply_velocity_and_angle(gas, cfg.velocity_scale, cfg.plume_angle_deg)

    regolith = RegolithProperties(
        grain_density=cfg.grain_density,
        gravity=cfg.gravity,
        cohesion_model=default_cohesion_model,
    )

    result = run_psi_model(gas, regolith, d_p=cfg.particle_diameter, c_e=cfg.c_e, compute_ejection=True)

    diameters = np.linspace(30e-6, 450e-6, 18)
    velocity_scales = np.linspace(0.1, 2.0, 16)
    plume_angles = np.linspace(10.0, 80.0, 15)

    sweep_d = sweep_particle_size(gas, regolith, diameters=diameters, c_e=cfg.c_e)
    sweep_v = sweep_velocity_scale(
        gas,
        regolith,
        d_p=cfg.particle_diameter,
        c_e=cfg.c_e,
        velocity_scales=velocity_scales,
    )
    sweep_a = sweep_plume_angle(
        gas,
        regolith,
        d_p=cfg.particle_diameter,
        c_e=cfg.c_e,
        plume_angles_deg=plume_angles,
    )

    sanity_checks(gas, regolith, d_p=cfg.particle_diameter, c_e=cfg.c_e)

    print("=== PSI Prototype Summary ===")
    print(f"Source: {source_tag}")
    print(f"Molecular mass [kg]: {gas.molecular_mass:.6e}")
    print(f"Patch points used: {gas.r.size}")
    print(f"Mean safety factor: {np.mean(result.safety_factor):.4f}")
    print(f"Lift fraction: {np.mean(result.lift_mask):.4f}")
    print(f"Integrated erosion: {_integrate_profile(result.erosion_rate, result.r):.6e}")
    print(f"Best plume-angle sweep erosion: {np.max(sweep_a['erosion_integral']):.6e}")

    if cfg.show_plots:
        make_plots(result, sweep_d, sweep_v, source_tag=source_tag)


if __name__ == "__main__":
    main()
