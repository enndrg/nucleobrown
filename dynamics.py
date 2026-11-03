from __future__ import annotations

from typing import Optional

import numpy as np

from .config import MediumConfig


def ou_noise(
    dt: np.ndarray, gamma: float, vth2: float, dim: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Bruit gaussien exact (sur v, sur x) d'un Ornstein-Uhlenbeck sur des intervalles ``dt``.

    Avec x = gamma*dt et e = exp(-x), pour chaque composante :
        Var(nv)      = vth2 (1 - e^2)
        Var(nx)      = vth2/gamma^2 (2x - 3 + 4e - e^2)
        Cov(nv, nx)  = vth2/gamma (1 - e)^2
    Un développement en série évite la perte de précision quand x << 1.
    """
    K = dt.size
    x = gamma * dt
    var_v = vth2 * (-np.expm1(-2.0 * x))
    series = (2.0 / 3.0) * x**3 - 0.5 * x**4 + (7.0 / 30.0) * x**5
    with np.errstate(over="ignore", invalid="ignore"):
        direct = 2.0 * x - 3.0 + 4.0 * np.exp(-x) - np.exp(-2.0 * x)
    f = np.where(x < 1e-2, series, direct)
    var_x = vth2 / gamma**2 * f
    cov = vth2 / gamma * np.expm1(-x) ** 2

    sv = np.sqrt(np.maximum(var_v, 0.0))
    ok = sv > 0
    a = np.where(ok, cov / np.where(ok, sv, 1.0), 0.0)
    b = np.sqrt(np.maximum(var_x - a**2, 0.0))
    z1 = rng.standard_normal((K, dim))
    z2 = rng.standard_normal((K, dim))
    nv = sv[:, None] * z1
    nx = a[:, None] * z1 + b[:, None] * z2
    return nv, nx


def damped_cumsum(
    t: np.ndarray, u: np.ndarray, gamma: float, v0: np.ndarray, span: float = 40.0
) -> np.ndarray:
    """Résout ``v_k = exp(-gamma (t_k - t_{k-1})) v_{k-1} + u_k`` avec v_{-1} = v0 en t_{-1} = 0.

    Solution : v_k = v0 e^{-gamma t_k} + sum_{j<=k} e^{-gamma (t_k - t_j)} u_j.
    On traite par blocs d'étendue temporelle ``span/gamma`` pour que les
    exponentielles ``e^{+gamma (t_j - t_0)}`` restent bornées par e^span.
    """
    K = t.size
    v = np.empty_like(u)
    carry = v0.astype(float).copy()
    t_ref = 0.0
    horizon = span / gamma
    start = 0
    while start < K:
        t0 = t[start]
        end = max(int(np.searchsorted(t, t0 + horizon, side="right")), start + 1)
        tb = t[start:end]
        c0 = carry * np.exp(-gamma * (t0 - t_ref))
        w = np.exp(gamma * (tb - t0))[:, None]
        s = np.cumsum(w * u[start:end], axis=0)
        vb = np.exp(-gamma * (tb - t0))[:, None] * (c0 + s)
        v[start:end] = vb
        carry = vb[-1]
        t_ref = tb[-1]
        start = end
    return v


def propagate(
    times: np.ndarray,
    impulses: np.ndarray,
    medium: MediumConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Propage la particule de t = 0 jusqu'aux instants ``times`` (triés).

    ``impulses[k]`` est appliquée à l'instant ``times[k]`` (nulle pour un simple
    point d'échantillonnage). Retourne ``(x, v, v0)`` : position à chaque
    instant, vitesse juste après le choc, vitesse initiale. En régime
    sur-amorti, ``v`` et ``v0`` valent ``None``.
    """
    K, dim = impulses.shape
    m, g, kT = medium.mass, medium.gamma, medium.kT
    dt = np.diff(times, prepend=0.0)

    if medium.regime == "overdamped":
        steps = impulses / (m * g)
        if kT > 0:
            d_th = kT / (m * g)
            steps = steps + rng.standard_normal((K, dim)) * np.sqrt(2.0 * d_th * dt)[:, None]
        return np.cumsum(steps, axis=0), None, None

    vth2 = kT / m
    v0 = np.sqrt(vth2) * rng.standard_normal(dim) if kT > 0 else np.zeros(dim)

    if g == 0.0:  # particule libre soumise aux seuls chocs : marche aléatoire en vitesse
        v = v0 + np.cumsum(impulses / m, axis=0)
        v_prev = np.vstack([v0, v[:-1]])
        x = np.cumsum(v_prev * dt[:, None], axis=0)
        return x, v, v0

    u = impulses / m
    nx: np.ndarray | float = 0.0
    if kT > 0:
        nv, nx = ou_noise(dt, g, vth2, dim, rng)
        u = u + nv
    v = damped_cumsum(times, u, g, v0)
    v_prev = np.vstack([v0, v[:-1]])
    coef = -np.expm1(-g * dt) / g
    x = np.cumsum(coef[:, None] * v_prev + nx, axis=0)
    return x, v, v0
