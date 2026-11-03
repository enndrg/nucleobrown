from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import expm

from .config import BETA, GAMMA, Config
from .isotopes import Isotope, resolve_chain
from .kicks import KickModel


def generator(chain: tuple[Isotope, ...]) -> np.ndarray:
    K = len(chain)
    Q = np.zeros((K + 1, K + 1))
    for s, iso in enumerate(chain):
        Q[s, s] = -iso.decay_rate
        Q[s, s + 1] = iso.decay_rate
    return Q


def stage_decayed_prob(chain: tuple[Isotope, ...], t: np.ndarray) -> np.ndarray:
    Q = generator(chain)
    K = len(chain)
    out = np.empty((t.size, K))
    for i, ti in enumerate(t):
        row = expm(Q * ti)[0]
        cum = np.cumsum(row[::-1])[::-1]  # cum[j] = P(état >= j)
        out[i] = cum[1:]
    return np.clip(out, 0.0, 1.0)


def expected_stage_events(cfg: Config, chain, t: np.ndarray) -> np.ndarray:
    src = cfg.source
    if src.mode == "batch":
        return src.n_nuclei * stage_decayed_prob(chain, t)
    t_max = float(np.max(t))
    t_small = min(iso.half_life for iso in chain) / 100.0
    parts = [np.linspace(0.0, t_max, 2001), t]
    if 0 < t_small < t_max:
        parts.append(np.geomspace(t_small, t_max, 1500))
    td = np.unique(np.concatenate(parts))
    cum = cumulative_trapezoid(stage_decayed_prob(chain, td), td, axis=0, initial=0.0)
    out = np.stack([np.interp(t, td, cum[:, s]) for s in range(cum.shape[1])], axis=1)
    return src.rate * out


def stage_emission_matrix(chain) -> np.ndarray:
    from .config import KIND_INDEX

    mat = np.zeros((len(chain), 3))
    for s, iso in enumerate(chain):
        names, probs = iso.emission_probs()
        for k, p in zip(names, probs):
            mat[s, KIND_INDEX[k]] += p
        mat[s, GAMMA] += iso.gamma_prob
    return mat


def stage_second_moments(cfg: Config, chain) -> np.ndarray:
    m2 = KickModel(cfg.kicks, cfg.medium.dim).second_moment()
    return stage_emission_matrix(chain) @ m2


def expected_counts(cfg: Config, t: np.ndarray) -> np.ndarray:
    chain = resolve_chain(cfg.source)
    return expected_stage_events(cfg, chain, t) @ stage_emission_matrix(chain)


def msd_theory(cfg: Config, t: np.ndarray):
    med = cfg.medium
    if med.regime != "overdamped":
        return None
    chain = resolve_chain(cfg.source)
    n_ev = expected_stage_events(cfg, chain, t)
    kicks = (n_ev @ stage_second_moments(cfg, chain)) / (med.mass * med.gamma) ** 2
    thermal = 2.0 * med.dim * (med.kT / (med.mass * med.gamma)) * t
    return kicks + thermal


def diffusion_theory(cfg: Config):
    med = cfg.medium
    if cfg.source.mode != "continuous" or med.gamma <= 0:
        return None
    chain = resolve_chain(cfg.source)
    d_kicks = cfg.source.rate * stage_second_moments(cfg, chain).sum() / (
        2.0 * med.dim * (med.mass * med.gamma) ** 2
    )
    d_th = med.kT / (med.mass * med.gamma)
    return dict(D_kicks=d_kicks, D_thermal=d_th, D_total=d_kicks + d_th)


def vacf_theory_steady(cfg: Config, lags: np.ndarray):
    med = cfg.medium
    if cfg.source.mode != "continuous" or med.regime != "underdamped" or med.gamma <= 0:
        return None
    chain = resolve_chain(cfg.source)
    shot = cfg.source.rate * stage_second_moments(cfg, chain).sum() / (med.mass**2 * 2.0 * med.gamma)
    c0 = shot + med.dim * med.kT / med.mass
    return c0 * np.exp(-med.gamma * lags)
