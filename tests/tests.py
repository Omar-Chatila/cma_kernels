import contextlib
import io
import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernelcma.factory import pure_brw_grouped, pure_cor_grouped
from kernelcma.postprocessing import clip_density_to_mass
from kernelcma.trajectories import build_state_trajectories


warnings.filterwarnings("ignore")

OUT_DIR = Path("tests/kernel_examples")
PICKLE_PATH = Path("tests/hmm_annotated.pickle")
STATE_COL = "state"
DT_TOLERANCE = 1.0
BASE_RANGE = 100
BASE_RESO = 2 * BASE_RANGE + 1

EXAMPLES = [
    {
        "name": "correlated_regularized_p99",
        "method": "correlated",
        "density_config": "regularized",
        "mass_percentile": 99,
    },
    {
        "name": "correlated_adaptive_p97",
        "method": "correlated",
        "density_config": "adaptive",
        "mass_percentile": 97,
    },
    {
        "name": "correlated_bayesian_p99",
        "method": "correlated",
        "density_config": "bayesian",
        "mass_percentile": 99,
    },
    {
        "name": "correlated_custom_gmm_tied_p95",
        "method": "correlated",
        "density_config": {
            "method": "gmm",
            "n_components": 2,
            "covariance_type": "tied",
            "reg_covar": 3.0,
            "random_state": 22,
        },
        "mass_percentile": 95,
    },
    {
        "name": "brownian_regularized_dt0p5_p99",
        "method": "brownian",
        "density_config": "regularized",
        "brownian_dt": 0.5,
        "mass_percentile": 99,
    },
    {
        "name": "brownian_terrain_dt1_p99",
        "method": "brownian",
        "density_config": "terrain",
        "brownian_dt": 1.0,
        "mass_percentile": 99,
    },
    {
        "name": "brownian_adaptive_dt0p25_p97",
        "method": "brownian",
        "density_config": "adaptive",
        "brownian_dt": 0.25,
        "mass_percentile": 97,
    },
    {
        "name": "brownian_bayesian_dt2_p95",
        "method": "brownian",
        "density_config": "bayesian",
        "brownian_dt": 2.0,
        "mass_percentile": 95,
    },
    {
        "name": "brownian_custom_bayesian_gmm_dt0p5_p99",
        "method": "brownian",
        "density_config": {
            "method": "bayesian_gmm",
            "n_components": 4,
            "covariance_type": "full",
            "reg_covar": 2.0,
            "weight_concentration_prior": 0.1,
            "random_state": 22,
        },
        "brownian_dt": 0.5,
        "mass_percentile": 99,
    },
]


def load_state_trajectories():
    traj_col = pickle.load(open(PICKLE_PATH, "rb"))
    gdf = traj_col.to_point_gdf()
    time_col = "timestamp" if "timestamp" in gdf.columns else "timestamp_utc"
    return build_state_trajectories(
        gdf,
        id_col=traj_col.get_traj_id_col(),
        time_col=time_col,
        geom_col="geometry",
        state_col=STATE_COL,
    )


def run_example(example, trajectories, dt_threshold, state_values):
    method = example["method"]
    mass_percentile = example["mass_percentile"]
    output_path = OUT_DIR / f"{example['name']}.png"
    title = (
        f"{PICKLE_PATH.stem} {STATE_COL} {method} | "
        f"{_density_label(example['density_config'])} | p{mass_percentile}"
    )

    kwargs = dict(
        dt_threshold=dt_threshold,
        dt_tolerance=DT_TOLERANCE,
        animal_trajectories=trajectories,
        rnge=BASE_RANGE,
        reso=BASE_RESO,
        out=str(output_path),
        state_values=state_values,
        density_config=example["density_config"],
        mass_percentile=mass_percentile,
        title=title,
    )

    if method == "brownian":
        kwargs["brownian_dt"] = example.get("brownian_dt", 1.0)
        densities = pure_brw_grouped(**kwargs)
    elif method == "correlated":
        densities = pure_cor_grouped(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    plt.close("all")
    return output_path, summarize_ranges(densities, state_values, mass_percentile)


def summarize_ranges(densities, state_values, mass_percentile):
    ranges = []
    for state_value, density in zip(state_values, densities):
        info = clip_density_to_mass(density, BASE_RANGE, mass_percentile)
        ranges.append(f"state {state_value}: r={info.rnge:.1f}, mass={info.retained_mass:.3f}")
    return "; ".join(ranges)


def _density_label(config):
    if isinstance(config, str):
        return config
    return f"{config['method']}_custom"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        trajectories, dt_threshold, state_values = load_state_trajectories()

    print(f"Generating {len(EXAMPLES)} kernel example plots in {OUT_DIR}")
    print(f"source={PICKLE_PATH}, state_col={STATE_COL}, dt_threshold={dt_threshold}")
    for example in EXAMPLES:
        with contextlib.redirect_stdout(io.StringIO()):
            output_path, range_summary = run_example(example, trajectories, dt_threshold, state_values)
        print(f"{output_path}: {range_summary}")


if __name__ == "__main__":
    main()
