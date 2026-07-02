from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Kernel2D:
    Z: Any
    rnge: float
    reso: int
    dx: float
    state_value: Any = None
    radius_cells: Optional[int] = None
    retained_mass: Optional[float] = None
    mass_percentile: Optional[float] = None
    dt_model_s: Optional[float] = None
