def fit_gaussian_density(axs, steps, rnge, reso, *, n_components=3):
    import numpy as np
    from sklearn.mixture import GaussianMixture

    data = np.array(steps)
    if len(data) < n_components:
        return None

    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=22)
    gmm.fit(data)

    x = np.linspace(-rnge, rnge, reso)
    y = np.linspace(-rnge, rnge, reso)
    X, Y = np.meshgrid(x, y)
    grid = np.column_stack([X.ravel(), Y.ravel()])

    log_density = gmm.score_samples(grid)
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
