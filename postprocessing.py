from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class KernelClip:
    Z: object
    rnge: float
    reso: int
    dx: float
    radius_cells: int
    retained_mass: float
    mass_percentile: Optional[float] = None


def normalize_mass_percentile(mass_percentile):
    if mass_percentile is None:
        return None
    value = float(mass_percentile)
    if value <= 0:
        raise ValueError("mass_percentile must be positive.")
    if value > 1:
        value /= 100.0
    if value > 1:
        raise ValueError("mass_percentile must be <= 1.0 or <= 100.")
    return value


def clip_density_to_mass(Z, rnge, mass_percentile=0.99):
    import numpy as np

    if Z is None:
        return KernelClip(None, rnge, 0, 0, 0, 0.0, normalize_mass_percentile(mass_percentile))

    density = np.asarray(Z)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("Kernel density must be a square 2D array.")

    reso = density.shape[0]
    dx = 2 * rnge / reso
    percentile = normalize_mass_percentile(mass_percentile)
    if percentile is None:
        return KernelClip(density, rnge, reso, dx, reso // 2, 1.0, None)

    total = float(np.nansum(density))
    if not np.isfinite(total) or total <= 0:
        return KernelClip(density, rnge, reso, dx, reso // 2, 0.0, percentile)

    center = reso // 2
    rows, cols = np.indices(density.shape)
    radii = np.sqrt((rows - center) ** 2 + (cols - center) ** 2)
    order = np.argsort(radii.ravel())
    cumulative = np.cumsum(density.ravel()[order])
    cutoff_index = int(np.searchsorted(cumulative, percentile * total, side="left"))
    cutoff_index = min(cutoff_index, len(order) - 1)
    radius_cells = int(np.ceil(radii.ravel()[order[cutoff_index]]))
    radius_cells = max(1, min(radius_cells, center))

    low = center - radius_cells
    high = center + radius_cells + 1
    clipped = density[low:high, low:high]
    retained_mass = float(np.nansum(clipped) / total)
    clipped_rnge = radius_cells * dx

    return KernelClip(
        clipped,
        clipped_rnge,
        clipped.shape[0],
        dx,
        radius_cells,
        retained_mass,
        percentile,
    )
