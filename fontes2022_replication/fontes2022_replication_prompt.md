# Development Prompt: Lunar Regolith Particle Trajectory Solver
### Replication of Fontes et al. (2022), *Acta Astronautica 195, 169–182*

---

## Task

Implement a Python particle trajectory solver replicating the regolith transport model from:

> Fontes, D., Mantovani, J.G., Metzger, P. (2022). *Numerical estimations of lunar regolith trajectories and damage potential due to rocket plumes.* Acta Astronautica, 195, 169–182. https://doi.org/10.1016/j.actaastro.2022.02.016

---

## 1. Project Structure

Deliver four files:

| File | Purpose |
|---|---|
| `particle_solver.py` | Main solver and physics |
| `postprocess.py` | Batch results loader and plot generator |
| `config.yaml` | All simulation parameters |
| `requirements.txt` | Pinned dependencies |

---

## 2. Input Data — OpenFOAM Reader

Use **`fluidfoam`** (`pip install fluidfoam`) to read the DSMC steady-state solution.

The reader must:
- Accept a path to an OpenFOAM case directory and a time directory (e.g. `"3"`) specified in `config.yaml`
- Read cell-centre coordinates using `fluidfoam.readmesh()`
- Read the following fields using `fluidfoam.readfield()`:
  - `U` — gas velocity vector field [m/s]
  - `rho` — gas density scalar field [kg/m³]
  - `T` — gas static temperature scalar field [K]
- Store all fields as NumPy arrays indexed by cell ID
- Store cell-centre coordinates as a single `(N_cells × 3)` NumPy array for use by the cell tracker

---

## 3. Cell-Tracking Algorithm

At every RK4 sub-step, locate the gas properties at the current particle position `x_p` using a **brute-force nearest-centroid search**, exactly as described in the paper (Section 3):

```python
def find_nearest_cell(x_p: np.ndarray, cell_centres: np.ndarray) -> int:
    """
    Sweep all mesh cells to find the index of the closest centroid
    to the particle position x_p. Replicates the tracking algorithm
    of Fontes et al. (2022) Section 3.

    NOTE: This is O(N_cells) per sub-step per particle and will be slow
    for large meshes. A scipy.spatial.KDTree is the recommended upgrade
    for production use.
    """
    deltas = cell_centres - x_p           # (N_cells x 3)
    dist2  = np.einsum('ij,ij->i', deltas, deltas)  # squared distances
    return int(np.argmin(dist2))
```

- Call this function at each of the four RK4 sub-steps (`k1`–`k4`) to retrieve `U[idx]`, `rho[idx]`, `T[idx]`
- Before calling the tracker, perform a bounding-box check against the domain limits in `config.yaml`
- If the particle is outside the bounding box, flag it as escaped and terminate integration immediately — do not call the cell tracker on out-of-bounds positions

---

## 4. Physics

Implement **drag force and weight only**, exactly as in the paper. All other forces are explicitly neglected.

### 4.1 Drag Force (Eq. 5)

$$F_{d,i} = m_p \frac{3 \rho C_D}{4 \rho_p d_p} \left| u_{g,i} - u_{p,i} \right| \left( u_{g,i} - u_{p,i} \right)$$

### 4.2 Drag Coefficient (Eq. 6) — Lane et al. empirical correlation

$$C_D = \begin{cases} 24.0 \, Re^{-1} & Re < 2 \\ 18.5 \, Re^{-0.6} & 2 \leq Re < 500 \\ 0.44 & Re \geq 500 \end{cases}$$

### 4.3 Particle Reynolds Number

$$Re_p = \frac{\rho \, \left| u_g - u_p \right| \, d_p}{A \, T^{\beta}}, \qquad A = 1.71575 \times 10^{-7}, \quad \beta = 0.78$$

where `T` is the local gas static temperature from the DSMC solution.

### 4.4 Combined Buoyancy–Weight Force (Eq. 7)

$$F_{w,i} = \left(1 - \frac{\rho}{\rho_p}\right) m_p \, g_i, \qquad g = 1.62 \ \text{m/s}^2 \ \text{(lunar gravity, downward)}$$

### 4.5 Neglected Forces

The following forces are explicitly **not** implemented:
- Lift force
- Virtual mass force
- Basset history force
- Electrostatic / dusty plasma forces

---

## 5. Numerical Integration

- Use a **4th-order Runge–Kutta (RK4)** scheme
- State vector at each step: `[x, y, z, ux, uy, uz]`
- The RK4 derivative function must call `find_nearest_cell()` at each of the four sub-steps to retrieve local gas properties at the current sub-step position
- Fixed time step read from `config.yaml` (paper default: `dt = 1e-4 s`)

**Termination conditions** (whichever occurs first):
1. Maximum simulation time `t_max` reached
2. Particle escapes the domain bounding box → status `"escaped"`
3. Particle grounds permanently (detected via repeated bottom-wall collisions with near-zero vertical velocity) → status `"grounded"`

**Bottom-wall elastic collision:** when `z ≤ 0`, reflect the wall-normal velocity component:

```
uz → -uz
```

then continue integration.

---

## 6. Batch Execution

Run a **full parameter sweep** over all combinations defined in `config.yaml`:

| Parameter | Values |
|---|---|
| Particle diameters | `[1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]` m |
| Initial radial positions | `np.arange(1.0, 10.5, 0.5)` m (19 values) |
| Initial height | `z_0 = 0.3 m` above ground (fixed) |
| Initial velocity | `u_p = 0` (at rest, fixed) |

Total cases: **9 × 19 = 171 particle simulations**.

For each case, store a results dictionary keyed by `(d_p, r_p0)` containing:

```python
{
    "trajectory": np.ndarray,   # shape (N_steps, 6): [x, y, z, ux, uy, uz]
    "exit_position": np.ndarray,  # (3,) at termination
    "exit_velocity": np.ndarray,  # (3,) at termination
    "exit_time": float,
    "status": str               # "escaped", "max_time", or "grounded"
}
```

Serialise the full results dictionary to a compressed NumPy archive on completion:

```python
np.savez_compressed(config["output"]["results_file"], **results)
```

---

## 7. Config File (`config.yaml`)

```yaml
openfoam:
  case_dir: "./dsmc_case"
  time_dir: "3"

domain:
  x_max: 20.0      # m
  y_max: 20.0      # m
  z_min: 0.0       # m  (ground surface)
  z_max: 10.0      # m

lander:
  mass_kg: 10525   # 11-ton Apollo-scale lander
  altitude_m: 1.0

particle:
  density_kg_m3: 3100.0        # lunar regolith bulk density
  initial_height_m: 0.3
  diameters_m:
    - 1.0e-6
    - 5.0e-6
    - 1.0e-5
    - 5.0e-5
    - 1.0e-4
    - 5.0e-4
    - 1.0e-3
    - 5.0e-3
    - 1.0e-2
  radial_positions_m:
    - 1.0
    - 1.5
    - 2.0
    - 2.5
    - 3.0
    - 3.5
    - 4.0
    - 4.5
    - 5.0
    - 5.5
    - 6.0
    - 6.5
    - 7.0
    - 7.5
    - 8.0
    - 8.5
    - 9.0
    - 9.5
    - 10.0

integration:
  dt_s: 1.0e-4
  t_max_s: 30.0

physics:
  gravity_m_s2: 1.62
  drag_A: 1.71575e-7
  drag_beta: 0.78

output:
  results_file: "results.npz"
  plots_dir: "./plots"
```

---

## 8. Post-processing and Plots (`postprocess.py`)

Load `results.npz` and generate the following three plots using exit-condition values for each particle. Style to match the paper: log axes where indicated, one line per radial position `r_p0`, colour-coded.

### Plot 1 — Velocity Magnitude vs. Particle Diameter
- x-axis: `d_p` [m], **log scale**
- y-axis: `|v_p|` [m/s], **log scale**
- One line per `r_p0`, colour-coded
- Filename: `velocity_vs_diameter.png`
- Replicates top-left panels of Figs. 14–17 in the paper

### Plot 2 — Ejection Angle vs. Particle Diameter
- x-axis: `d_p` [m], **log scale**
- y-axis: angle [deg] = `arctan2(v_z, v_x)` at domain exit, **linear scale**
- One line per `r_p0`, colour-coded
- Filename: `angle_vs_diameter.png`
- Replicates top-right panels of Figs. 14–17 in the paper

### Plot 3 — Kinetic Energy vs. Particle Diameter
- x-axis: `d_p` [m], **log scale**
- y-axis: `KE = 0.5 * m_p * |v_p|^2` [J], **log scale**
- One line per `r_p0`, colour-coded
- Filename: `kinetic_energy_vs_diameter.png`
- Replicates bottom-left panels of Figs. 14–17 in the paper

Save all plots as `.png` to the `plots_dir` specified in `config.yaml`. Use **`matplotlib` only** — no additional plotting libraries.

---

## 9. Dependencies (`requirements.txt`)

```
numpy>=1.24
scipy>=1.10
pyyaml>=6.0
matplotlib>=3.7
fluidfoam>=0.2
```

---

## 10. Code Quality Requirements

### Required standalone functions in `particle_solver.py`

| Function | References |
|---|---|
| `compute_reynolds()` | Eq. after Eq. 6, Section 3 |
| `compute_cd()` | Eq. 6, Section 3 |
| `compute_drag()` | Eq. 5, Section 3 |
| `compute_weight()` | Eq. 7, Section 3 |
| `rk4_step()` | Section 3 |
| `find_nearest_cell()` | Section 3, paper tracking algorithm |

All functions must be:
- **Standalone and unit-testable** (no hidden global state)
- Annotated with **type hints** on all arguments and return values
- Documented with **docstrings** that reference the relevant equation numbers from Fontes et al. (2022)

### Additional requirements

- `find_nearest_cell()` must include a comment flagging the O(N_cells) brute-force cost and recommending a `scipy.spatial.KDTree` upgrade for production use
- `particle_solver.py` must include a `if __name__ == "__main__":` guard
- Print a progress line to stdout for each completed case in the following format:

```
Completed dp=1.00e-06 m | r0=1.0 m | status=escaped | |v|=312.4 m/s
```

---

## Reference

Fontes, D., Mantovani, J.G., Metzger, P. (2022).
*Numerical estimations of lunar regolith trajectories and damage potential due to rocket plumes.*
Acta Astronautica, **195**, 169–182.
https://doi.org/10.1016/j.actaastro.2022.02.016
