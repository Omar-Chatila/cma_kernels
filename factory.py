from .plotting import create_and_plot_kernels
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

    def get_state_kernels(self, dt_tolerance, rnge, reso, out=None):
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
        )


def state_kernels_from_trajectories(
    animal_trajectories,
    dt_threshold,
    state_values,
    dt_tolerance,
    rnge,
    reso,
    out=None,
):
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
    )
    brw_densities = pure_brw_grouped(
        dt_threshold,
        dt_tolerance,
        animal_trajectories,
        rnge,
        reso,
        out,
        state_values=state_values,
    )

    correlated = tuple(
        Kernel2D(Z, rnge, reso, dx, state_value)
        for Z, state_value in zip(cor_densities, state_values)
    )
    brownian = tuple(
        Kernel2D(Z, rnge, reso, dx, state_value)
        for Z, state_value in zip(brw_densities, state_values)
    )

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
):
    if state_values is None and num_states is not None:
        state_values = list(range(1, num_states + 1))
    steps = calculate_steps_cor_grouped(dt_threshold, dt_tolerance, animal_trajectories, state_values)
    return create_and_plot_kernels(steps, rnge, reso, out, state_values=state_values)


def pure_brw_grouped(
    dt_threshold,
    dt_tolerance,
    animal_trajectories,
    rnge,
    reso,
    out=None,
    num_states=None,
    state_values=None,
):
    if state_values is None and num_states is not None:
        state_values = list(range(1, num_states + 1))
    steps = calculate_steps_brownian_grouped(dt_threshold, dt_tolerance, animal_trajectories, state_values)
    return create_and_plot_kernels(steps, rnge, reso, out, state_values=state_values)
