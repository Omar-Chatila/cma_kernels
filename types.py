from dataclasses import dataclass
from typing import Any


@dataclass
class Kernel2D:
    Z: Any
    rnge: float
    reso: int
    dx: float
    state_value: Any = None
