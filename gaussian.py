from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from math import inf


DENSITY_PRESETS = {
    "legacy": {
        "method": "gmm",
        "n_components": 3,
        "covariance_type": "full",
        "reg_covar": 1e-6,
        "random_state": 22,
        "min_steps": 3,
    },
    "regularized": {
        "method": "gmm",
        "n_components": 3,
        "covariance_type": "tied",
        "reg_covar": 2.0,
        "random_state": 22,
        "min_steps": 3,
    },
    "adaptive": {
        "method": "gmm",
        "n_components": "auto",
        "component_thresholds": ((80, 1), (250, 2), (inf, 3)),
        "covariance_type": "tied",
        "reg_covar": 2.0,
        "random_state": 22,
        "min_steps": 3,
    },
    "terrain": {
        "method": "gmm",
        "n_components": "auto",
        "component_thresholds": ((80, 1), (250, 2), (inf, 3)),
        "covariance_type": "tied",
        "reg_covar": 4.0,
        "random_state": 22,
        "min_steps": 10,
    },
    "bayesian": {
        "method": "bayesian_gmm",
        "n_components": 3,
        "covariance_type": "full",
        "reg_covar": 2.0,
        "random_state": 22,
        "min_steps": 3,
        "weight_concentration_prior": 0.1,
    },
}

_METHOD_ALIASES = {
    "gmm": "gmm",
    "gaussian_mixture": "gmm",
    "gaussianmixture": "gmm",
    "GaussianMixture": "gmm",
    "bayesian": "bayesian_gmm",
    "bayesian_gmm": "bayesian_gmm",
    "bayesiangaussianmixture": "bayesian_gmm",
    "BayesianGaussianMixture": "bayesian_gmm",
}


class DensityConfigurationError(ValueError):
    """Raised when a density estimator configuration cannot be used."""


def resolve_density_config(config=None, **overrides):
    """Return a complete density-estimator configuration.

    ``config`` can be ``None``, a preset name, a mapping, or a scikit-learn
    style estimator object with ``fit`` and ``score_samples``.  Keyword
    arguments override preset values; ``None`` override values are ignored.
    """
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    if "reg_covariance" in clean_overrides and "reg_covar" not in clean_overrides:
        clean_overrides["reg_covar"] = clean_overrides.pop("reg_covariance")
    override_preset = clean_overrides.pop("preset", None)

    if config is None:
        resolved = _preset_config(override_preset) if override_preset is not None else dict(DENSITY_PRESETS["legacy"])
    elif isinstance(config, str):
        resolved = _preset_config(override_preset or config)
    elif isinstance(config, Mapping):
        preset = override_preset if override_preset is not None else config.get("preset")
        resolved = _preset_config(preset) if preset is not None else dict(DENSITY_PRESETS["legacy"])
        resolved.update({key: value for key, value in config.items() if key != "preset" and value is not None})
    elif hasattr(config, "fit") and hasattr(config, "score_samples"):
        resolved = dict(DENSITY_PRESETS["legacy"])
        resolved["model"] = config
    else:
        raise DensityConfigurationError(
            "density config must be None, a preset name, a mapping, or an estimator with fit/score_samples"
        )

    resolved.update(clean_overrides)
    if "reg_covariance" in resolved and "reg_covar" not in resolved:
        resolved["reg_covar"] = resolved.pop("reg_covariance")
    return resolved


def fit_gaussian_density(
    axs,
    steps,
    rnge,
    reso,
    *,
    config=None,
    preset=None,
    method=None,
    model=None,
    n_components=None,
    covariance_type=None,
    reg_covar=None,
    reg_covariance=None,
    random_state=None,
    min_steps=None,
    component_thresholds=None,
    **model_kwargs,
):
    import numpy as np

    options = resolve_density_config(
        config,
        preset=preset,
        method=method,
        model=model,
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        reg_covariance=reg_covariance,
        random_state=random_state,
        min_steps=min_steps,
        component_thresholds=component_thresholds,
        **model_kwargs,
    )

    data = np.array(steps)
    if len(data) < int(options.get("min_steps", 3)):
        return None

    estimator = _make_estimator(options, len(data))
    if estimator is None:
        return None
    estimator.fit(data)

    x = np.linspace(-rnge, rnge, reso)
    y = np.linspace(-rnge, rnge, reso)
    X, Y = np.meshgrid(x, y)
    grid = np.column_stack([X.ravel(), Y.ravel()])

    log_density = estimator.score_samples(grid)
    density = np.exp(log_density)
    Z = density.reshape(X.shape)

    if axs is not None:
        axs.imshow(
            Z,
            extent=(-rnge, rnge, -rnge, rnge),
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
        )
    return Z


def _preset_config(name):
    if name not in DENSITY_PRESETS:
        choices = ", ".join(sorted(DENSITY_PRESETS))
        raise DensityConfigurationError(f"unknown density preset {name!r}; choose from: {choices}")
    return dict(DENSITY_PRESETS[name])


def _component_count(options, sample_count):
    configured = options.get("n_components", 3)
    if callable(configured):
        configured = configured(sample_count)
    elif configured in (None, "auto"):
        thresholds = options.get("component_thresholds") or DENSITY_PRESETS["adaptive"]["component_thresholds"]
        configured = next(count for limit, count in thresholds if sample_count < limit)
    configured = int(configured)
    if configured < 1:
        return None
    return min(configured, sample_count)


def _make_estimator(options, sample_count):
    model = options.get("model")
    if model is not None:
        return _clone_estimator(model)

    n_components = _component_count(options, sample_count)
    if n_components is None:
        return None

    method = _METHOD_ALIASES.get(str(options.get("method", "gmm")), str(options.get("method", "gmm")).lower())
    kwargs = _estimator_kwargs(options)
    kwargs["n_components"] = n_components

    if method == "gmm":
        from sklearn.mixture import GaussianMixture

        return GaussianMixture(**kwargs)
    if method == "bayesian_gmm":
        from sklearn.mixture import BayesianGaussianMixture

        return BayesianGaussianMixture(**kwargs)
    raise DensityConfigurationError(f"unknown density method {options.get('method')!r}")


def _estimator_kwargs(options):
    reserved = {
        "preset",
        "method",
        "model",
        "n_components",
        "min_steps",
        "component_thresholds",
    }
    return {key: value for key, value in options.items() if key not in reserved and value is not None}


def _clone_estimator(model):
    try:
        from sklearn.base import clone

        return clone(model)
    except Exception:
        return deepcopy(model)
