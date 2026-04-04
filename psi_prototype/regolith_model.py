from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class RegolithProperties:
    grain_density: float
    gravity: float
    cohesion_model: Optional[Callable[[np.ndarray], np.ndarray]] = None
    cohesion_constant: float = 0.0

    def cohesion_force(self, d_p: np.ndarray) -> np.ndarray:
        if self.cohesion_model is not None:
            return self.cohesion_model(d_p)
        return np.full_like(d_p, self.cohesion_constant, dtype=float)


def default_cohesion_model(d_p: np.ndarray) -> np.ndarray:
    # Simple inverse-size trend often used for fine regolith cohesion scaling.
    d_ref = 100e-6
    f_ref = 3.0e-8
    return f_ref * np.sqrt(d_ref / np.maximum(d_p, 1e-12))


def grain_projected_area(d_p: np.ndarray) -> np.ndarray:
    return 0.25 * np.pi * d_p ** 2


def grain_weight_force(d_p: np.ndarray, rho_p: float, g: float) -> np.ndarray:
    return (np.pi / 6.0) * d_p ** 3 * rho_p * g
