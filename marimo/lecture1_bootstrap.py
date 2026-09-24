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
        # Bootstrap Hypothesis Testing

        Suppose we observe two samples *A* and *B* and want to know whether their means
        differ significantly — but we do not want to assume a specific distribution.

        **The bootstrap (permutation) approach:**
        1. Compute the *observed* difference in means.
        2. Under the null hypothesis (H₀: A and B come from the same distribution),
           any labelling of observations as "A" or "B" is equally likely.
        3. Combine A and B, then randomly relabel them thousands of times, recording the
           difference in means each time. This builds the *null distribution*.
        4. The **p-value** is the fraction of null-distribution differences that are
           at least as extreme as the observed difference.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## Set up the two samples")
    return


@app.cell
def _(mo):
    delta_slider = mo.ui.slider(
        0.80, 1.00, value=0.95, step=0.01,
        label="Effect size δ  (sample B = A × δ, so δ < 1 means B is lower)"
    )
    size_slider = mo.ui.slider(
        10, 200, value=50, step=10, label="Sample size (n per group)"
    )
    n_boot_slider = mo.ui.slider(
        1000, 20000, value=8000, step=1000, label="Bootstrap replications"
    )
    mo.vstack([delta_slider, size_slider, n_boot_slider])
    return delta_slider, n_boot_slider, size_slider


@app.cell
def _(delta_slider, n_boot_slider, np, plt, size_slider):
    delta = delta_slider.value
    n = size_slider.value
    N_boot = n_boot_slider.value

    # Generate samples: A ~ N(10, 1), B = A * delta
    rng = np.random.default_rng()
    A = 10.0 + rng.normal(0, 1, n)
    B = A * delta

    observed_diff = A.mean() - B.mean()

    # Build null distribution via permutation (no-replacement split of combined sample)
    AB = np.concatenate([A, B])
    # Vectorised permutation: argsort of uniform noise gives random ordering
    perm_indices = np.argsort(rng.uniform(size=(N_boot, 2 * n)), axis=1)
    boot_diff = AB[perm_indices[:, :n]].mean(axis=1) - AB[perm_indices[:, n:]].mean(axis=1)

    # Two-sided p-value
    p_value = float(np.mean(np.abs(boot_diff) >= np.abs(observed_diff)))

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: the two raw samples
    bins_shared = np.linspace(min(A.min(), B.min()) - 0.5, max(A.max(), B.max()) + 0.5, 30)
    axes[0].hist(A, bins=bins_shared, alpha=0.6, color="steelblue", label="Sample A")
    axes[0].hist(B, bins=bins_shared, alpha=0.6, color="coral", label="Sample B")
    axes[0].axvline(A.mean(), color="steelblue", lw=2, linestyle="--",
                    label=f"Mean A = {A.mean():.3f}")
    axes[0].axvline(B.mean(), color="coral", lw=2, linestyle="--",
                    label=f"Mean B = {B.mean():.3f}")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Raw samples (δ = {delta:.2f})\nObserved difference = {observed_diff:.4f}")
    axes[0].legend(fontsize=8)

    # Right: bootstrap null distribution
    axes[1].hist(boot_diff, bins=60, density=True, alpha=0.7, color="silver",
                 label="Bootstrap differences (H₀)")
    axes[1].axvline(observed_diff, color="red", lw=2,
                    label=f"Observed diff = {observed_diff:.4f}")
    axes[1].axvline(-observed_diff, color="red", lw=2, linestyle="--",
                    label=f"Mirror = {-observed_diff:.4f}")
    tail_color = "red"
    tail_mask_pos = boot_diff >= abs(observed_diff)
    tail_mask_neg = boot_diff <= -abs(observed_diff)
    if tail_mask_pos.any():
        axes[1].hist(boot_diff[tail_mask_pos], bins=60, density=True,
                     alpha=0.5, color=tail_color)
    if tail_mask_neg.any():
        axes[1].hist(boot_diff[tail_mask_neg], bins=60, density=True,
                     alpha=0.5, color=tail_color)
    axes[1].set_xlabel("Difference in means (A − B) under H₀")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"Bootstrap null distribution\np-value = {p_value:.4f}")
    axes[1].legend(fontsize=8)

    fig
    return A, B, N_boot, boot_diff, delta, n, observed_diff, p_value


@app.cell
def _(mo, delta, n, observed_diff, p_value):
    significance = (
        f"**significant** at the 5% level (p = {p_value:.4f} < 0.05)"
        if p_value < 0.05
        else f"**not significant** at the 5% level (p = {p_value:.4f} ≥ 0.05)"
    )
    kind = "success" if p_value < 0.05 else "warn"
    mo.callout(
        mo.md(
            f"""
            **Results with n = {n} and δ = {delta:.2f}:**

            - Observed difference in means: **{observed_diff:.4f}**
            - Bootstrap p-value (two-sided): **{p_value:.4f}**
            - Conclusion: the difference is {significance}.

            The red shaded region is the fraction of bootstrap differences that exceed
            the observed value in absolute size — that fraction *is* the p-value.

            Try dragging δ towards 1.0 to make the effect smaller and watch the p-value
            rise above 0.05 (the difference becomes undetectable).
            """
        ),
        kind=kind,
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ---
        ## How does this compare to a t-test?

        The classical Student's t-test also tests whether two means differ. It assumes both
        samples are drawn from a normal distribution and uses an analytical formula for the
        test statistic. The bootstrap makes **no distributional assumption** — it only requires
        that the sample is representative of the population.

        For large samples from roughly normal data, the two approaches give very similar
        p-values. The bootstrap is more flexible when the normality assumption fails.
        """
    )
    return


if __name__ == "__main__":
    app.run()
