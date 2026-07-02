from .plotting import create_and_plot_kernels
from .postprocessing import clip_density_to_mass
from .steps import calculate_steps_brownian_grouped, calculate_steps_cor_grouped
from .trajectories import build_state_trajectories
from .types import Kernel2D


class StateKernelFactory:
    def __init__(
        self,
        gdf,
        *,
        id_col="individual-local-identifier",
        time_col="timestamp",
        geom_col="geometry",
        state_col="state",
    ):
        self.gdf = gdf
        self.id_col = id_col
        self.time_col = time_col
        self.geom_col = geom_col
        self.state_col = state_col
        self.trajectories = None
        self.dt_threshold = None
        self.state_values = None

    def build_trajectories(self):
        self.trajectories, self.dt_threshold, self.state_values = build_state_trajectories(
            self.gdf,
            id_col=self.id_col,
            time_col=self.time_col,
            geom_col=self.geom_col,
            state_col=self.state_col,
        )
        return self.trajectories, self.dt_threshold, self.state_values

    def get_state_kernels(
        self,
        dt_tolerance,
        rnge,
        reso,
        out=None,
        density_config=None,
        brownian_dt=1.0,
        dt_model_s=None,
        mass_percentile=0.99,
        **density_kwargs,
    ):
        if self.trajectories is None:
            self.build_trajectories()
        return state_kernels_from_trajectories(
            self.trajectories,
            self.dt_threshold,
            self.state_values,
            dt_tolerance,
            rnge,
            reso,
            out,
            density_config=density_config,
            brownian_dt=brownian_dt,
            dt_model_s=dt_model_s,
            mass_percentile=mass_percentile,
            **density_kwargs,
        )


def state_kernels_from_trajectories(
    animal_trajectories,
    dt_threshold,
    state_values,
    dt_tolerance,
    rnge,
    reso,
    out=None,
    density_config=None,
    brownian_dt=1.0,
    dt_model_s=None,
    mass_percentile=0.99,
    **density_kwargs,
):
    brownian_dt = _resolve_brownian_dt(brownian_dt, dt_model_s)
    dt_model_s = None if brownian_dt is None else float(brownian_dt) * 60.0
    dx = 2 * rnge / reso
    print(f"dx: {dx}\n")

    cor_densities = pure_cor_grouped(
        dt_threshold,
        dt_tolerance,
        animal_trajectories,
        rnge,
        reso,
        out,
        state_values=state_values,
        density_config=density_config,
        mass_percentile=mass_percentile,
        **density_kwargs,
    )
    brw_densities = pure_brw_grouped(
        dt_threshold,
        dt_tolerance,
        animal_trajectories,
        rnge,
        reso,
        out,
        state_values=state_values,
        density_config=density_config,
        brownian_dt=brownian_dt,
        mass_percentile=mass_percentile,
        **density_kwargs,
    )

    correlated = _kernel_tuple(cor_densities, state_values, rnge, mass_percentile)
    brownian = _kernel_tuple(brw_densities, state_values, rnge, mass_percentile, dt_model_s=dt_model_s)

    return correlated, brownian


def pure_cor_grouped(
    dt_threshold,
    dt_tolerance,
    animal_trajectories,
    rnge,
    reso,
    out=None,
    num_states=None,
    state_values=None,
    density_config=None,
    mass_percentile=0.99,
    **density_kwargs,
):
    if state_values is None and num_states is not None:
        state_values = list(range(1, num_states + 1))
    steps = calculate_steps_cor_grouped(dt_threshold, dt_tolerance, animal_trajectories, state_values)
    return create_and_plot_kernels(
        steps,
        rnge,
        reso,
        out,
        state_values=state_values,
        density_config=density_config,
        mass_percentile=mass_percentile,
        **density_kwargs,
    )


def pure_brw_grouped(
    dt_threshold,
    dt_tolerance,
    animal_trajectories,
    rnge,
    reso,
    out=None,
    num_states=None,
    state_values=None,
    density_config=None,
    brownian_dt=1.0,
    dt_model_s=None,
    mass_percentile=0.99,
    **density_kwargs,
):
    if state_values is None and num_states is not None:
        state_values = list(range(1, num_states + 1))
    brownian_dt = _resolve_brownian_dt(brownian_dt, dt_model_s)
    steps = calculate_steps_brownian_grouped(
        dt_threshold,
        dt_tolerance,
        animal_trajectories,
        state_values,
        brownian_dt=brownian_dt,
    )
    return create_and_plot_kernels(
        steps,
        rnge,
        reso,
        out,
        state_values=state_values,
        density_config=density_config,
        mass_percentile=mass_percentile,
        **density_kwargs,
    )


def _resolve_brownian_dt(brownian_dt, dt_model_s):
    if dt_model_s is not None:
        return float(dt_model_s) / 60.0
    return brownian_dt


def _kernel_tuple(densities, state_values, rnge, mass_percentile, dt_model_s=None):
    kernels = []
    for Z, state_value in zip(densities, state_values):
        info = clip_density_to_mass(Z, rnge, mass_percentile)
        kernels.append(
            Kernel2D(
                info.Z,
                info.rnge,
                info.reso,
                info.dx,
                state_value,
                radius_cells=info.radius_cells,
                retained_mass=info.retained_mass,
                mass_percentile=info.mass_percentile,
                dt_model_s=dt_model_s,
            )
        )
    return tuple(kernels)
