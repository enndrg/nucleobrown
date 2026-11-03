from __future__ import annotations

import os
from pathlib import Path

import matplotlib

if not os.environ.get("NUCLEOBROWN_GUI"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from .config import KINDS, Config

COLORS = {"alpha": "#d62728", "beta": "#1f77b4", "gamma": "#2ca02c"}


def plot_trajectory(showcase: dict, cfg: Config):
    dim = cfg.medium.dim
    t = np.concatenate([[0.0], showcase["times"]])
    p = np.vstack([np.zeros((1, dim)), showcase["pos"]])
    fig = plt.figure(figsize=(13, 5.5))
    if dim == 3:
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.plot(p[:, 0], p[:, 1], p[:, 2], lw=0.5)
        ax.scatter(*p[0], c="g")
        ax.scatter(*p[-1], c="r")
    elif dim == 2:
        ax = fig.add_subplot(1, 2, 1)
        pts = p.reshape(-1, 1, 2)
        lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap="viridis", linewidths=0.7)
        lc.set_array(t[1:])
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_aspect("equal")
        fig.colorbar(lc, ax=ax, label="temps (s)")
        ax.plot(*p[0], "go", label="départ")
        ax.plot(*p[-1], "ro", label="arrivée")
        ax.legend()
    else:
        ax = fig.add_subplot(1, 2, 1)
        ax.step(t, p[:, 0], where="post", lw=0.8)
        ax.set_xlabel("temps (s)")
        ax.set_ylabel("x")
    ax.set_title(f"Trajectoire ({t.size - 1} chocs, régime {cfg.medium.regime})")

    ax2 = fig.add_subplot(1, 2, 2)
    for i, k in enumerate(KINDS):
        tk = showcase["times"][showcase["kinds"] == i]
        ax2.step(np.concatenate([[0.0], tk]), np.arange(tk.size + 1), where="post", color=COLORS[k], label=k)
    ax2.set_xlabel("temps (s)")
    ax2.set_ylabel("émissions cumulées")
    ax2.set_title("Émissions par type")
    ax2.legend()
    fig.tight_layout()
    return fig


def plot_msd(arrays: dict, summary: dict, cfg: Config):
    t, msd, sem = arrays["t"], arrays["msd"], arrays["sem"]
    s = t > 0
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(t[s], msd[s], label="simulation")
    a1.fill_between(t[s], (msd - 2 * sem)[s], (msd + 2 * sem)[s], alpha=0.25, label="±2 SEM")
    if arrays.get("msd_theory") is not None:
        a1.plot(t[s], arrays["msd_theory"][s], "k--", label="théorie exacte")
    tf, yf = arrays["fit_curve"]
    a1.plot(tf, yf, "r-", lw=2.5, alpha=0.7, label=f"ajust. α = {summary['msd_fit']['alpha']:.2f}")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("t (s)")
    a1.set_ylabel(r"$\langle r^2 \rangle$")
    a1.set_title("Déplacement quadratique moyen")
    a1.legend()
    a1.grid(alpha=0.3, which="both")

    a2.semilogx(t[s], arrays["local_alpha"][s])
    a2.axhline(1.0, color="k", ls=":", label="diffusion normale (α=1)")
    a2.set_xlabel("t (s)")
    a2.set_ylabel(r"$d\ln\langle r^2\rangle / d\ln t$")
    a2.set_title("Exposant local")
    a2.legend()
    a2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_distributions(ens, arrays: dict, summary: dict, cfg: Config):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for snap in summary["snapshots"]:
        x = ens.pos[:, snap["index"], :].ravel()
        if x.std() == 0:
            continue
        a1.hist(x / x.std(), bins=40, density=True, histtype="step", label=f"t = {snap['t']:.3g} s")
    u = np.linspace(-4, 4, 300)
    a1.plot(u, np.exp(-u**2 / 2) / np.sqrt(2 * np.pi), "k--", label="gaussienne N(0,1)")
    a1.set_xlabel("déplacement standardisé (par composante)")
    a1.set_ylabel("densité")
    a1.set_title("Distribution des déplacements")
    a1.legend()
    t = arrays["t"]
    a2.semilogx(t[1:], arrays["alpha2"][1:])
    a2.axhline(0, color="k", ls=":")
    a2.set_xlabel("t (s)")
    a2.set_ylabel(r"$\alpha_2(t)$")
    a2.set_title("Paramètre de non-gaussianité")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_activity(arrays: dict, summary: dict, cfg: Config):
    t = arrays["t"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for i, k in enumerate(KINDS):
        a1.plot(t, arrays["counts_mean"][:, i], color=COLORS[k], label=f"{k} (sim.)")
        a1.plot(t, arrays["counts_theory"][:, i], "--", color="k", lw=1)
    a1.plot([], [], "k--", label="théorie")
    a1.set_xlabel("t (s)")
    a1.set_ylabel("émissions cumulées moyennes")
    a1.set_title("Émissions")
    a1.legend()
    a2.plot(t[1:], arrays["fano"], ".", ms=3)
    a2.axhline(1.0, color="k", ls=":", label="Poisson (F = 1)")
    pooled = summary["emissions"]["fano_pooled"]
    a2.axhline(pooled, color="r", label=f"F groupé = {pooled:.3f}")
    a2.set_xlabel("t (s)")
    a2.set_ylabel("facteur de Fano des comptages")
    a2.set_title("Statistique des désintégrations")
    a2.legend()
    fig.tight_layout()
    return fig


def plot_vacf(arrays: dict, cfg: Config):
    lags, c = arrays["vacf"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lags, c, label="simulation")
    if arrays.get("vacf_theory") is not None:
        ax.plot(lags, arrays["vacf_theory"], "k--", label="théorie stationnaire")
    ax.set_xlabel("délai τ (s)")
    ax.set_ylabel(r"$\langle v(0)\cdot v(\tau)\rangle$")
    ax.set_title("Autocorrélation des vitesses")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def save_all(ens, summary: dict, arrays: dict, cfg: Config, outdir: Path) -> list[Path]:
    outdir = Path(outdir)
    figs = {}
    if ens.showcase is not None:
        figs["trajectory.png"] = plot_trajectory(ens.showcase, cfg)
    figs["msd.png"] = plot_msd(arrays, summary, cfg)
    figs["distributions.png"] = plot_distributions(ens, arrays, summary, cfg)
    figs["activity.png"] = plot_activity(arrays, summary, cfg)
    if "vacf" in arrays:
        figs["vacf.png"] = plot_vacf(arrays, cfg)
    paths = []
    for name, fig in figs.items():
        p = outdir / name
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_sweep(param: str, values, d_fit, d_err, d_th, alpha):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    v = np.asarray(values, float)
    a1.errorbar(v, d_fit, yerr=d_err, fmt="o", capsize=3, label="D ajusté")
    if all(x is not None for x in d_th):
        a1.plot(v, d_th, "k--", label="D théorique")
    a1.set_xlabel(param)
    a1.set_ylabel("D")
    if np.all(v > 0):
        a1.set_xscale("log")
    if np.all(np.asarray(d_fit) > 0):
        a1.set_yscale("log")
    a1.legend()
    a1.grid(alpha=0.3, which="both")
    a2.plot(v, alpha, "s-")
    a2.axhline(1.0, color="k", ls=":")
    a2.set_xlabel(param)
    a2.set_ylabel("exposant α")
    if np.all(v > 0):
        a2.set_xscale("log")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    return fig
