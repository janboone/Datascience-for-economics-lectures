# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _(mo):
    mo.md(
        """
        # Distribution of an Estimator

        Statistical estimators — like the sample mean or an OLS slope — are themselves random
        variables. Each time you draw a new sample, you get a slightly different estimate.
        **Statistical hacking** means we study these distributions not with analytical formulas,
        but by running many simulations and observing the resulting histogram.

        This app has two parts:
        1. **Sample mean distribution** — illustrating the Central Limit Theorem
        2. **OLS slope distribution** — showing how regression estimates vary across samples
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## Part 1: Distribution of the Sample Mean")
    return


@app.cell
def _(mo):
    mo.md(
        """
        We draw many samples of size *n* from the uniform distribution on [0, 1] and compute
        the mean of each sample. The histogram of these means shows the *sampling distribution*
        of the estimator.

        **Theory:** for large *n*, the Central Limit Theorem guarantees this distribution
        approaches a normal with mean 0.5 and standard error SE = 1/√(12·n).
        """
    )
    return


@app.cell
def _(mo):
    sample_size_slider = mo.ui.slider(
        2, 200, value=10, step=2, label="Sample size (n)"
    )
    n_sim_slider = mo.ui.slider(
        500, 15000, value=5000, step=500, label="Number of simulations"
    )
    mo.hstack([sample_size_slider, n_sim_slider], justify="start", gap=2)
    return n_sim_slider, sample_size_slider


@app.cell
def _(n_sim_slider, np, plt, sample_size_slider):
    n = sample_size_slider.value
    N = n_sim_slider.value

    rng1 = np.random.default_rng()
    samples = rng1.uniform(0, 1, (N, n))
    means = samples.mean(axis=1)

    theoretical_se = 1.0 / np.sqrt(12.0 * n)
    x_range = np.linspace(max(0, 0.5 - 4 * theoretical_se), min(1, 0.5 + 4 * theoretical_se), 300)
    normal_pdf = (
        (1.0 / (theoretical_se * np.sqrt(2 * np.pi)))
        * np.exp(-0.5 * ((x_range - 0.5) / theoretical_se) ** 2)
    )

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.hist(means, bins=60, density=True, alpha=0.65, color="steelblue",
             label=f"Simulated means (n={n}, N={N})")
    ax1.plot(x_range, normal_pdf, "r-", lw=2,
             label=f"Theoretical N(0.5, SE={theoretical_se:.4f})")
    ax1.set_xlabel("Sample mean")
    ax1.set_ylabel("Density")
    ax1.set_title("Sampling distribution of the mean")
    ax1.legend()
    fig1
    return fig1, means, n, N, theoretical_se


@app.cell
def _(mo, n, theoretical_se):
    mo.callout(
        mo.md(
            f"""
            With n = **{n}** observations per sample:
            - Theoretical standard error: **SE = 1/√(12·{n}) = {theoretical_se:.4f}**
            - The distribution is centred on 0.5 (the true population mean).
            - Increase *n* to see the distribution narrow — this is the Central Limit Theorem.
            - The red curve is the theoretical normal; it matches the histogram even for moderate *n*.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md("---\n## Part 2: Distribution of the OLS Slope Estimator")
    return


@app.cell
def _(mo):
    mo.md(
        """
        We repeatedly draw samples, fit a simple OLS regression y = β₀ + β₁x + ε,
        and record the estimated slope β̂₁. The histogram shows how much the slope estimate
        varies across samples — and therefore how uncertain our estimate is.

        The right panel shows 50 fitted regression lines drawn from this distribution.
        Notice how the uncertainty is largest far from the mean of *x*.
        """
    )
    return


@app.cell
def _(mo):
    true_slope_slider = mo.ui.slider(
        -2.0, 2.0, value=0.5, step=0.1, label="True slope (β₁)"
    )
    noise_slider = mo.ui.slider(
        0.1, 3.0, value=0.5, step=0.1, label="Noise level (σ)"
    )
    n_ols_slider = mo.ui.slider(
        5, 100, value=20, step=5, label="Sample size per regression"
    )
    n_ols_sim_slider = mo.ui.slider(
        200, 5000, value=1000, step=200, label="Number of regressions"
    )
    mo.vstack([
        mo.hstack([true_slope_slider, noise_slider], justify="start", gap=2),
        mo.hstack([n_ols_slider, n_ols_sim_slider], justify="start", gap=2),
    ])
    return n_ols_sim_slider, n_ols_slider, noise_slider, true_slope_slider


@app.cell
def _(n_ols_sim_slider, n_ols_slider, noise_slider, np, plt, true_slope_slider):
    true_slope = true_slope_slider.value
    sigma = noise_slider.value
    n_ols = n_ols_slider.value
    N_ols = n_ols_sim_slider.value
    true_intercept = 1.0

    rng2 = np.random.default_rng()
    x_mat = rng2.normal(0, 1, (N_ols, n_ols))
    eps_mat = rng2.normal(0, sigma, (N_ols, n_ols))
    y_mat = true_intercept + true_slope * x_mat + eps_mat

    # Vectorised OLS: slope = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
    x_c = x_mat - x_mat.mean(axis=1, keepdims=True)
    y_c = y_mat - y_mat.mean(axis=1, keepdims=True)
    slopes_hat = (x_c * y_c).sum(axis=1) / (x_c ** 2).sum(axis=1)
    intercepts_hat = y_mat.mean(axis=1) - slopes_hat * x_mat.mean(axis=1)

    # Theoretical SE of slope: σ / sqrt(n · Var(x)) ≈ σ / sqrt(n) for std-normal x
    slope_se = sigma / np.sqrt(n_ols)
    x_plot = np.linspace(-3, 3, 100)
    normal_slope_pdf = (
        (1.0 / (slope_se * np.sqrt(2 * np.pi)))
        * np.exp(-0.5 * ((x_plot - true_slope) / slope_se) ** 2)
    )

    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of estimated slopes
    axes[0].hist(slopes_hat, bins=50, density=True, alpha=0.65, color="coral",
                 label="Estimated slopes β̂₁")
    axes[0].axvline(true_slope, color="k", lw=2, linestyle="--",
                    label=f"True slope = {true_slope:.1f}")
    x_hist_range = np.linspace(slopes_hat.min() - 0.2, slopes_hat.max() + 0.2, 300)
    normal_hist_pdf = (
        (1.0 / (slope_se * np.sqrt(2 * np.pi)))
        * np.exp(-0.5 * ((x_hist_range - true_slope) / slope_se) ** 2)
    )
    axes[0].plot(x_hist_range, normal_hist_pdf, "b-", lw=2,
                 label=f"Theoretical N({true_slope:.1f}, {slope_se:.3f}²)")
    axes[0].set_xlabel("Estimated slope β̂₁")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution of OLS slope estimator")
    axes[0].legend(fontsize=8)

    # Fan of 50 regression lines
    axes[1].set_xlim(-3, 3)
    y_true_line = true_intercept + true_slope * x_plot
    for i in range(min(50, N_ols)):
        axes[1].plot(x_plot, intercepts_hat[i] + slopes_hat[i] * x_plot,
                     color="coral", alpha=0.15, lw=1)
    axes[1].plot(x_plot, y_true_line, "k--", lw=2, label="True regression line")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("50 estimated regression lines")
    axes[1].legend()

    fig2
    return fig2, slope_se, slopes_hat, intercepts_hat


@app.cell
def _(mo, slope_se, true_slope):
    mo.callout(
        mo.md(
            f"""
            With these settings, the OLS slope estimate has:
            - **Mean:** {true_slope:.2f} (the estimator is unbiased — it centres on the true value)
            - **Standard error:** σ/√n ≈ **{slope_se:.4f}** (smaller n → more spread)
            - The fan of lines is widest far from x = 0 because small errors in slope are amplified.
            - Increase the noise σ or decrease n to see greater uncertainty.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()
