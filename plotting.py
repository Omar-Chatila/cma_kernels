from .gaussian import fit_gaussian_density

updating = False


def generate_heatmap(axs, coords, rnge, reso):
    import numpy as np

    coords = np.array(coords)
    x_edges = np.linspace(-rnge, rnge, reso)
    y_edges = np.linspace(-rnge, rnge, reso)

    heatmap, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=[x_edges, y_edges])
    axs.imshow(
        heatmap.T,
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        origin="lower",
        cmap="viridis",
    )


def create_and_plot_kernels(state_steps, rnge, reso, output_dir=None, state_values=None):
    import matplotlib.pyplot as plt

    state_values = list(range(len(state_steps))) if state_values is None else list(state_values)
    state_count = len(state_steps)
    if state_count == 0:
        return []

    fig, axs = plt.subplots(2, state_count, figsize=(4 * state_count, 6), squeeze=False)
    densities = []
    linked_pairs = []

    for idx, steps in enumerate(state_steps):
        axs[0, idx].set_title(f"State {state_values[idx]} steps")
        axs[1, idx].set_title(f"State {state_values[idx]} density")

        density = None
        if len(steps) >= 3:
            generate_heatmap(axs[0, idx], steps, rnge, reso)
            density = fit_gaussian_density(axs[1, idx], steps, rnge, reso)
        densities.append(density)
        linked_pairs.append((axs[0, idx], axs[1, idx]))

    def on_xlim_changed(event_ax):
        global updating
        if updating:
            return

        updating = True
        for ax1, ax2 in linked_pairs:
            if event_ax == ax1:
                ax2.set_xlim(ax1.get_xlim())
                ax2.set_ylim(ax1.get_ylim())
            elif event_ax == ax2:
                ax1.set_xlim(ax2.get_xlim())
                ax1.set_ylim(ax2.get_ylim())

        fig.canvas.draw_idle()
        updating = False

    for ax1, ax2 in linked_pairs:
        ax1.callbacks.connect("xlim_changed", on_xlim_changed)
        ax1.callbacks.connect("ylim_changed", on_xlim_changed)
        ax2.callbacks.connect("xlim_changed", on_xlim_changed)
        ax2.callbacks.connect("ylim_changed", on_xlim_changed)

    if output_dir is not None:
        plt.savefig(output_dir, bbox_inches="tight")
    return densities
