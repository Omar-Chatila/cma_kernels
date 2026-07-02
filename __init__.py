"""Build movement kernels from state-annotated trajectories."""

from importlib import import_module

_EXPORTS = {
    "Kernel2D": "types",
    "StateKernelFactory": "factory",
    "build_state_trajectories": "trajectories",
    "calculate_steps_brownian_grouped": "steps",
    "calculate_steps_cor_grouped": "steps",
    "create_and_plot_kernels": "plotting",
    "DENSITY_PRESETS": "gaussian",
    "DensityConfigurationError": "gaussian",
    "fit_gaussian_density": "gaussian",
    "resolve_density_config": "gaussian",
    "generate_heatmap": "plotting",
    "infer_state_values": "trajectories",
    "pure_brw_grouped": "factory",
    "pure_cor_grouped": "factory",
    "rotate_vector": "steps",
    "state_kernels_from_trajectories": "factory",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.{_EXPORTS[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value
