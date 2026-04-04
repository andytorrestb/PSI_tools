from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

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


@dataclass
class TrajectoryPath:
    t: np.ndarray
    r: np.ndarray
    z: np.ndarray
    source_r: float
    landing_r: float
    flight_time: float
    max_height: float


@dataclass
class TrajectoryEnsemble:
    paths: List[TrajectoryPath]
    landing_r: np.ndarray
    flight_time: np.ndarray
    max_height: np.ndarray


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


def _integrate_single_trajectory(
    source_r: float,
    speed: float,
    angle_deg: float,
    gravity: float,
    dt: float,
    max_time: float,
    linear_drag: float,
) -> TrajectoryPath:
    angle = np.deg2rad(angle_deg)
    vr = speed * np.cos(angle)
    vz = speed * np.sin(angle)

    t_list = [0.0]
    r_list = [source_r]
    z_list = [0.0]

    t = 0.0
    r = source_r
    z = 0.0
    while t < max_time:
        ar = -linear_drag * vr
        az = -gravity - linear_drag * vz

        vr += ar * dt
        vz += az * dt
        r += vr * dt
        z += vz * dt
        t += dt

        t_list.append(t)
        r_list.append(r)
        z_list.append(z)

        if z <= 0.0 and t > 0.0:
            break

    t_arr = np.asarray(t_list, dtype=float)
    r_arr = np.asarray(r_list, dtype=float)
    z_arr = np.asarray(z_list, dtype=float)

    return TrajectoryPath(
        t=t_arr,
        r=r_arr,
        z=z_arr,
        source_r=float(source_r),
        landing_r=float(r_arr[-1]),
        flight_time=float(t_arr[-1]),
        max_height=float(np.max(z_arr)),
    )


def simulate_trajectory_ensemble(
    result: PsiResults,
    gravity: float,
    n_trajectories: int = 30,
    dt: float = 2.5e-4,
    max_time: float = 8.0,
    linear_drag: float = 0.0,
) -> TrajectoryEnsemble:
    lift_idx = np.where(result.lift_mask & (result.ejection_speed > 0.0))[0]
    if lift_idx.size == 0:
        return TrajectoryEnsemble(paths=[], landing_r=np.array([]), flight_time=np.array([]), max_height=np.array([]))

    if lift_idx.size > n_trajectories:
        pick = np.linspace(0, lift_idx.size - 1, n_trajectories).astype(int)
        lift_idx = lift_idx[pick]

    paths: List[TrajectoryPath] = []
    for idx in lift_idx:
        p = _integrate_single_trajectory(
            source_r=float(result.r[idx]),
            speed=float(result.ejection_speed[idx]),
            angle_deg=float(result.ejection_angle_deg[idx]),
            gravity=gravity,
            dt=dt,
            max_time=max_time,
            linear_drag=linear_drag,
        )
        paths.append(p)

    landing_r = np.asarray([p.landing_r for p in paths], dtype=float)
    flight_time = np.asarray([p.flight_time for p in paths], dtype=float)
    max_height = np.asarray([p.max_height for p in paths], dtype=float)
    return TrajectoryEnsemble(paths=paths, landing_r=landing_r, flight_time=flight_time, max_height=max_height)
