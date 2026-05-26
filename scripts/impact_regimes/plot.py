"""
Plot mean impact and price variance for every successfully-run regime.

For each regime, both quantities are shown on linear and log-log axes. The
mean impact carries error bars (SEM = sqrt(Var/N)) and a theoretical overlay.
On each log-log axis we fit a power-law y = A t^alpha by least squares on
log(t)-log(y) and print the estimated exponent in the panel title and stdout.

Pure-noise mean (E[p_t] ~ 0) has no meaningful power-law: that panel is
skipped with a note.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np

from config import D, HURST, J, RESULTS_DIR, RESULTS_FILE


def power_law_fit(t: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """Fit y ≈ A t^alpha by least-squares on log(t)-log(y); return (alpha, A)."""
    mask = (t > 0) & np.isfinite(y) & (y > 0)
    if mask.sum() < 3:
        return None
    slope, intercept = np.polyfit(np.log(t[mask]), np.log(y[mask]), 1)
    return float(slope), float(np.exp(intercept))


def theoretical_impact(r0: float, r1: float, m0: float, m1: float, t: np.ndarray) -> np.ndarray:
    """Pick the appropriate theoretical mean-impact prediction by regime."""
    if r0 == 0:
        return np.zeros_like(t)
    I_low = r0 * np.sqrt(D * t / np.pi)
    if r0 <= 1:
        return I_low
    I_high = np.sqrt(2 * m0 * t / (J / D))  # = sqrt(2 m0 t / L)
    if r1 > r0:
        return np.sqrt(m0 / m1) * I_high
    return I_high


def _plot_linear(ax, t, y, sem, theory, label, ylabel, title):
    if sem is not None:
        ax.errorbar(t, y, yerr=sem, fmt="o-", ms=3, lw=1, capsize=2, label=label)
    else:
        ax.plot(t, y, "o-", ms=3, lw=1, label=label)
    if theory is not None:
        ax.plot(t, theory, "k--", lw=1, label="theory")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("$t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)


def _plot_loglog(ax, t, y, label, ylabel, title_prefix, name, quantity_key):
    positive = (t > 0) & np.isfinite(y) & (y > 0)
    if positive.sum() < 3:
        ax.text(0.5, 0.5, "no positive samples\n(fit n/a)",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title(f"{title_prefix} log-log (n/a)")
        ax.set_xlabel("$t$")
        ax.set_ylabel(ylabel)
        return

    t_pos, y_pos = t[positive], y[positive]
    ax.loglog(t_pos, y_pos, "o", ms=3, label=label)
    fit = power_law_fit(t_pos, y_pos)
    if fit is not None:
        alpha, A = fit
        ax.loglog(t_pos, A * t_pos ** alpha, "r--", lw=1.5,
                  label=fr"fit: $\alpha = {alpha:.3f}$")
        print(f"  {name:<22s} {quantity_key:<15s} alpha = {alpha:+.3f}")
        ax.set_title(fr"{title_prefix} log-log — $\alpha = {alpha:.3f}$")
    else:
        ax.set_title(f"{title_prefix} log-log (fit n/a)")
    ax.set_xlabel("$t$")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


def plot_impact_grid(regimes: dict) -> plt.Figure:
    n = len(regimes)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.3 * n), squeeze=False)
    fig.suptitle(r"Mean price impact $\mathbb{E}[p_t]$ — linear & log-log with power-law fit",
                 y=1.0)

    for row, (name, res) in enumerate(regimes.items()):
        r0, r1 = res["r0"], res["r1"]
        m0, m1 = r0 * J, r1 * J
        t = np.asarray(res["time"])
        y = np.asarray(res["price_mean"])
        sem = np.sqrt(np.asarray(res["price_variance"]) / res["N_samples"])
        theory = theoretical_impact(r0, r1, m0, m1, t)

        prefix = f"{name}: $r_0={r0}$, $r_1={r1}$"
        _plot_linear(axes[row, 0], t, y, sem, theory,
                     label=r"$\mathbb{E}[p_t]$ (MC $\pm$ SEM)",
                     ylabel="impact", title=f"{prefix} — linear")
        _plot_loglog(axes[row, 1], t, y, label="MC",
                     ylabel="impact", title_prefix=prefix,
                     name=name, quantity_key="impact")

    fig.tight_layout()
    return fig


def plot_variance_grid(regimes: dict) -> plt.Figure:
    n = len(regimes)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.3 * n), squeeze=False)
    fig.suptitle(r"Price variance $\mathrm{Var}(p_t)$ — linear & log-log with power-law fit",
                 y=1.0)

    for row, (name, res) in enumerate(regimes.items()):
        r0, r1 = res["r0"], res["r1"]
        t = np.asarray(res["time"])
        y = np.asarray(res["price_variance"])

        prefix = f"{name}: $r_0={r0}$, $r_1={r1}$"
        _plot_linear(axes[row, 0], t, y, sem=None, theory=None,
                     label=r"$\mathrm{Var}(p_t)$ (MC)",
                     ylabel="variance", title=f"{prefix} — linear")
        _plot_loglog(axes[row, 1], t, y, label="MC",
                     ylabel="variance", title_prefix=prefix,
                     name=name, quantity_key="variance")

    fig.tight_layout()
    return fig


def main() -> None:
    with open(RESULTS_FILE, "rb") as f:
        data = pickle.load(f)
    regimes = data["regimes"]
    failures = data["failures"]

    if failures:
        print("Regimes that failed (excluded from plots):")
        for name, msg in failures.items():
            print(f"  - {name}: {msg}")
    if not regimes:
        raise SystemExit("No successful regimes to plot.")

    print(f"\nExpected power-law exponents: impact ~ 0.5,  variance ~ {2*HURST-1:.2f}")
    print("\n=== Power-law fits ===")
    fig_impact = plot_impact_grid(regimes)
    fig_var = plot_variance_grid(regimes)

    fig_impact.savefig(RESULTS_DIR / "fig1_impact.png", dpi=130, bbox_inches="tight")
    fig_var.savefig(RESULTS_DIR / "fig2_variance.png", dpi=130, bbox_inches="tight")
    print(f"\nSaved figures to {RESULTS_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
