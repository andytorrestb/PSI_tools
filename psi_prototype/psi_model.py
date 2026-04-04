from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from gas_model import GasSurfaceData
from regolith_model import RegolithProperties, grain_projected_area, grain_weight_force


def _integrate_profile(y: np.ndarray, x: np.ndarray) -> float:
    # NumPy compatibility across versions where trapz may be removed.
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


@dataclass
class PsiResults:
    r: np.ndarray
    d_p: float
    f_n: np.ndarray
    f_t: np.ndarray
    f_resist: np.ndarray
    lift_mask: np.ndarray
    safety_factor: np.ndarray
    p_lift: np.ndarray
    erosion_rate: np.ndarray
    ejection_speed: np.ndarray
    ejection_angle_deg: np.ndarray


def aerodynamic_forces(gas: GasSurfaceData, d_p: float) -> Tuple[np.ndarray, np.ndarray]:
    a_p = grain_projected_area(np.asarray(d_p, dtype=float))
    # DSMC-consistent momentum flux force model from impact statistics.
    f_n = gas.phi * gas.molecular_mass * gas.vn ** 2 * a_p
    f_t = gas.phi * gas.molecular_mass * gas.vt * gas.vn * a_p
    return f_n, f_t


def evaluate_lift(
    gas: GasSurfaceData,
    regolith: RegolithProperties,
    d_p: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f_n, f_t = aerodynamic_forces(gas, d_p)
    f_g = grain_weight_force(np.asarray(d_p, dtype=float), regolith.grain_density, regolith.gravity)
    f_c = regolith.cohesion_force(np.full_like(gas.r, d_p, dtype=float))
    f_resist = f_g + f_c
    f_drive = f_t + f_n
    safety = np.divide(f_drive, f_resist, out=np.zeros_like(f_drive), where=f_resist > 0.0)
    lift = f_drive > f_resist
    p_lift = lift.astype(float)
    return f_n, f_t, f_resist, lift, safety, p_lift


def erosion_rate(phi: np.ndarray, p_lift: np.ndarray, c_e: float) -> np.ndarray:
    return c_e * phi * p_lift


def ejection_kinematics(gas: GasSurfaceData, lift_mask: np.ndarray, speed_factor: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    speed = np.zeros_like(gas.r)
    speed[lift_mask] = speed_factor * gas.vt[lift_mask]
    angle = np.rad2deg(np.arctan2(np.maximum(gas.vn, 1e-12), np.maximum(gas.vt, 1e-12)))
    return speed, angle


def run_psi_model(
    gas: GasSurfaceData,
    regolith: RegolithProperties,
    d_p: float,
    c_e: float,
    compute_ejection: bool = True,
) -> PsiResults:
    f_n, f_t, f_resist, lift, safety, p_lift = evaluate_lift(gas, regolith, d_p)
    mdot_e = erosion_rate(gas.phi, p_lift, c_e)
    if compute_ejection:
        ej_speed, ej_angle = ejection_kinematics(gas, lift)
    else:
        ej_speed = np.zeros_like(gas.r)
        ej_angle = np.zeros_like(gas.r)

    return PsiResults(
        r=gas.r,
        d_p=d_p,
        f_n=f_n,
        f_t=f_t,
        f_resist=f_resist,
        lift_mask=lift,
        safety_factor=safety,
        p_lift=p_lift,
        erosion_rate=mdot_e,
        ejection_speed=ej_speed,
        ejection_angle_deg=ej_angle,
    )


def sweep_particle_size(
    gas: GasSurfaceData,
    regolith: RegolithProperties,
    diameters: np.ndarray,
    c_e: float,
) -> Dict[str, np.ndarray]:
    erosion_integral = np.zeros_like(diameters, dtype=float)
    for i, d_p in enumerate(diameters):
        res = run_psi_model(gas, regolith, d_p=float(d_p), c_e=c_e)
        erosion_integral[i] = _integrate_profile(res.erosion_rate, res.r)
    return {"d_p": diameters, "erosion_integral": erosion_integral}


def sweep_velocity_scale(
    gas: GasSurfaceData,
    regolith: RegolithProperties,
    d_p: float,
    c_e: float,
    velocity_scales: np.ndarray,
) -> Dict[str, np.ndarray]:
    erosion_integral = np.zeros_like(velocity_scales, dtype=float)
    for i, s in enumerate(velocity_scales):
        gas_s = GasSurfaceData(
            r=gas.r,
            number_density=gas.number_density,
            rho=gas.rho,
            phi=gas.number_density * np.maximum(gas.vn * s, 0.0),
            vn=np.maximum(gas.vn * s, 0.0),
            vt=np.maximum(gas.vt * s, 0.0),
            velocity=gas.velocity * s,
            normal=gas.normal,
            molecular_mass=gas.molecular_mass,
        )
        res = run_psi_model(gas_s, regolith, d_p=d_p, c_e=c_e)
        erosion_integral[i] = _integrate_profile(res.erosion_rate, res.r)
    return {"velocity_scale": velocity_scales, "erosion_integral": erosion_integral}


def sweep_plume_angle(
    gas: GasSurfaceData,
    regolith: RegolithProperties,
    d_p: float,
    c_e: float,
    plume_angles_deg: np.ndarray,
) -> Dict[str, np.ndarray]:
    erosion_integral = np.zeros_like(plume_angles_deg, dtype=float)
    speed = np.sqrt(gas.vn ** 2 + gas.vt ** 2)
    for i, angle_deg in enumerate(plume_angles_deg):
        ang = np.deg2rad(angle_deg)
        vn = np.maximum(speed * np.cos(ang), 0.0)
        vt = np.maximum(speed * np.sin(ang), 0.0)
        gas_s = GasSurfaceData(
            r=gas.r,
            number_density=gas.number_density,
            rho=gas.rho,
            phi=gas.number_density * vn,
            vn=vn,
            vt=vt,
            velocity=gas.velocity,
            normal=gas.normal,
            molecular_mass=gas.molecular_mass,
        )
        res = run_psi_model(gas_s, regolith, d_p=d_p, c_e=c_e)
        erosion_integral[i] = _integrate_profile(res.erosion_rate, res.r)
    return {"plume_angle_deg": plume_angles_deg, "erosion_integral": erosion_integral}


def design_time_dependent_erosion() -> None:
    """Extension hook for transient bed evolution coupling.

    Intended future model:
    1. Update local bed height using erosion_rate * dt.
    2. Recompute local flow incidence and shielding effects.
    3. March in time with optional adaptive dt.
    """
    raise NotImplementedError("Design stub: transient erosion coupling not implemented.")


def design_secondary_particle_transport() -> None:
    """Extension hook for secondary particle trajectories.

    Intended future model:
    1. Sample ejection velocity and angle from local gas-grain forcing.
    2. Integrate ballistic or drag-coupled trajectories.
    3. Deposit/re-impact particles and update bed source terms.
    """
    raise NotImplementedError("Design stub: secondary transport not implemented.")
