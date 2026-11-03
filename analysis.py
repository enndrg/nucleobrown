from __future__ import annotations

import numpy as np
from scipy import stats

from . import theory
from .config import KINDS, Config
from .isotopes import resolve_chain
# MSD := <deplacement ^2>

def squared_displacement(pos: np.ndarray) -> np.ndarray:
    """|x(t) - x(0)|^2, forme (M, G+1)."""
    d = pos - pos[:, :1, :]
    return np.sum(d * d, axis=-1)


def window_mask(t: np.ndarray, frac, t_max: float) -> np.ndarray:
    mask = (t >= frac[0] * t_max) & (t <= frac[1] * t_max) & (t > 0)
    if mask.sum() < 3:
        raise ValueError("fenêtre d'ajustement trop étroite (moins de 3 points)")
    return mask


def fit_power_law(t, msd, mask):
    """MSD ~A t^alpha par régression log-log."""
    m = mask & (msd > 0)
    slope, intercept = np.polyfit(np.log(t[m]), np.log(msd[m]), 1)
    return float(slope), float(np.exp(intercept))
    #Retourne (alpha, A)

def fit_diffusion(t, msd, dim, mask):
    """ MSD = 2 d D t + c par régression linéaire. """
    slope, intercept = np.polyfit(t[mask], msd[mask], 1)
    return float(slope / (2.0 * dim)), float(intercept)
#Retourne (D, c)

def local_exponent(t: np.ndarray, msd: np.ndarray, smooth: int = 7) -> np.ndarray:
    """d log MSD / d log t (log-MSD lissé par moyenne glissante de largeur."""
    out = np.full(msd.shape, np.nan)
    ok = (t > 0) & (msd > 0)
    if ok.sum() >= 3:
        lm = np.log(msd[ok])
        if smooth > 1 and lm.size > smooth:
            pad = smooth // 2
            lm = np.convolve(np.pad(lm, pad, mode="edge"), np.ones(smooth) / smooth, mode="valid")
        out[ok] = np.gradient(lm, np.log(t[ok]))
    return out


def bootstrap_fits(r2, t, mask, dim, n_boot, rng):
    """Rééchantillonnage des trajectoires -> distribution de alpha et D."""
    M = r2.shape[0]
    al, ds = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        msd_b = r2[rng.integers(0, M, M)].mean(axis=0)
        al[b] = fit_power_law(t, msd_b, mask)[0]
        ds[b] = fit_diffusion(t, msd_b, dim, mask)[0]
    return al, ds


def tamsd(pos: np.ndarray, lag: int) -> np.ndarray:
    #MSD moyenné en temps (grille régulière) pour chaque trajectoire, (M,)
    d = pos[:, lag:, :] - pos[:, :-lag, :]
    return np.mean(np.sum(d * d, axis=-1), axis=1)


def vacf(vel: np.ndarray, max_lag: int, start: int = 0) -> np.ndarray:
    #corela6 des vitesses, moyennée sur trajectoires et origines
    v = vel[:, start:, :]
    n = v.shape[1]
    return np.array([np.mean(np.sum(v[:, : n - l, :] * v[:, l:, :], axis=-1)) for l in range(max_lag + 1)])


def non_gaussian_parameter(r2: np.ndarray, dim: int) -> np.ndarray:
    #alpha2 = d <r^4> / ((d+2) <r^2>^2) - 1 
    m2 = r2.mean(axis=0)
    m4 = (r2**2).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = dim * m4 / ((dim + 2) * m2**2) - 1.0
    a[m2 == 0] = np.nan
    return a


def fano_per_interval(primary: np.ndarray):
    #Facteur de Fano (variance/moyenne) des énumération par intervalle de grille
    dn = np.diff(primary, axis=1).astype(float)
    mean = dn.mean(axis=0)
    var = dn.var(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(mean > 0, var / np.where(mean > 0, mean, 1.0), np.nan)
    pooled = float(var.sum() / mean.sum()) if mean.sum() > 0 else float("nan")
    return f, pooled


def is_linear_grid(t: np.ndarray) -> bool:
    return bool(np.allclose(np.diff(t), t[1] - t[0], rtol=1e-9, atol=0))


# ---------------------------------------------------------------- analyse complète
def analyze(ens, cfg: Config):
#return (sommaire,array)
    t, pos = ens.grid, ens.pos
    M, _, dim = pos.shape
    G = t.size - 1
    rng = np.random.default_rng(cfg.run.seed + 1)

    r2 = squared_displacement(pos)
    msd = r2.mean(axis=0)
    sem = r2.std(axis=0, ddof=1) / np.sqrt(M)
    mask = window_mask(t, cfg.analysis.fit_window, cfg.run.t_max)

    alpha, amp = fit_power_law(t, msd, mask)
    D, c0 = fit_diffusion(t, msd, dim, mask)
    fit = dict(alpha=alpha, amplitude=amp, D=D, intercept=c0,
               window=[float(t[mask][0]), float(t[mask][-1])])
    if cfg.analysis.n_boot > 0:
        al_b, d_b = bootstrap_fits(r2, t, mask, dim, cfg.analysis.n_boot, rng)
        fit["alpha_std"] = float(al_b.std(ddof=1))
        fit["alpha_ci95"] = [float(x) for x in np.percentile(al_b, [2.5, 97.5])]
        fit["D_std"] = float(d_b.std(ddof=1))
        fit["D_ci95"] = [float(x) for x in np.percentile(d_b, [2.5, 97.5])]

    summary: dict = dict(
        n_runs=M, dim=dim, regime=cfg.medium.regime, mode=cfg.source.mode,
        chain=[iso.name for iso in resolve_chain(cfg.source)], msd_fit=fit,
    )
    arrays: dict = dict(t=t, msd=msd, sem=sem, local_alpha=local_exponent(t, msd), mask=mask)
    arrays["fit_curve"] = (t[mask], amp * t[mask] ** alpha)

#comparaison th-exp
    th_msd = theory.msd_theory(cfg, t)
    arrays["msd_theory"] = th_msd
    if th_msd is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (msd[1:] - th_msd[1:]) / sem[1:]
        z = z[np.isfinite(z)]
        summary["msd_vs_theory"] = dict(
            mean_z=float(z.mean()), max_abs_z=float(np.abs(z).max()),
            frac_within_3sigma=float(np.mean(np.abs(z) < 3)),
        )
    d_th = theory.diffusion_theory(cfg)
    if d_th is not None:
        d_th["relative_error_fit"] = float((D - d_th["D_total"]) / d_th["D_total"])
        if "D_std" in fit and fit["D_std"] > 0:
            d_th["z_fit"] = float((D - d_th["D_total"]) / fit["D_std"])
        summary["diffusion_theory"] = d_th

# ca marche pas sans jsp trop quoi ecrire c pour moy tempo = moy spatiale
    if is_linear_grid(t) and G >= 20:
        lag = max(1, G // 10)
        ta = tamsd(pos, lag)
        ens_val = float(r2[:, lag].mean()) if lag < r2.shape[1] else float("nan")
        summary["ergodicity"] = dict(
            lag_time=float(t[lag]), tamsd_mean=float(ta.mean()),
            eb_parameter=float(ta.var(ddof=1) / ta.mean() ** 2) if ta.mean() > 0 else None,
            ensemble_msd_at_lag=ens_val,
        )

# pas gauss
    arrays["alpha2"] = non_gaussian_parameter(r2, dim)
    snaps = []
    for f in cfg.analysis.snapshot_fracs:
        i = int(np.abs(t - f * cfg.run.t_max).argmin())
        x = pos[:, i, :].ravel()
        entry = dict(t=float(t[i]), index=i, alpha2=float(arrays["alpha2"][i]))
        if x.size >= 8 and x.std() > 0:
            entry["normaltest_p"] = float(stats.normaltest(x).pvalue)
            entry["skew"] = float(stats.skew(x))
            entry["excess_kurtosis"] = float(stats.kurtosis(x))
        snaps.append(entry)
    summary["snapshots"] = snaps

# stat
    fano, pooled = fano_per_interval(ens.primary)
    arrays["fano"] = fano
    arrays["counts_mean"] = ens.counts.mean(axis=0)
    arrays["counts_theory"] = theory.expected_counts(cfg, t)
    summary["emissions"] = dict(
        fano_pooled=pooled,
        mean_total={k: float(ens.counts[:, -1, i].mean()) for i, k in enumerate(KINDS)},
        theory_total={k: float(arrays["counts_theory"][-1, i]) for i, k in enumerate(KINDS)},
    )

# vitesses
    if ens.vel is not None and is_linear_grid(t) and G >= 20:
        start = G // 2
        max_lag = min((G + 1 - start) // 2, G // 4)
        lags = t[: max_lag + 1]
        arrays["vacf"] = (lags, vacf(ens.vel, max_lag, start=start))
        arrays["vacf_theory"] = theory.vacf_theory_steady(cfg, lags)
    return summary, arrays
