from __future__ import annotations

import numpy as np

from .config import BETA, KickConfig

# |E[B^2] pour B ~ Beta(2, 3) : a(a+1) / ((a+b)(a+b+1)) = 6 / 30
_BETA23_M2 = 2.0 * 3.0 / (5.0 * 6.0)


class KickModel:
    """impulsion : isotrope, ||.|| dépendant du type de particule

    alpha, gamma : ||.|| fixe. beta : ||.|| fixe, ou ``p_beta * Beta(2, 3)`` si
    ``beta_spectrum`` est actif (spectre continu en énergie du rayonnement beta)
    """

    def __init__(self, cfg: KickConfig, dim: int) -> None:
        self.cfg = cfg
        self.dim = dim
        self.p = np.array([cfg.p_alpha, cfg.p_beta, cfg.p_gamma], dtype=float)

    def sample(self, kinds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = kinds.size
        mag = self.p[kinds].copy()
        if self.cfg.beta_spectrum:
            m = kinds == BETA
            k = int(m.sum())
            if k:
                mag[m] *= rng.beta(2.0, 3.0, k)
        dirs = rng.standard_normal((n, self.dim))
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mag[:, None] * dirs / norms

    def second_moment(self) -> np.ndarray:
        # |E[|p|^2] pour chaque (alpha, beta, gamma)
        m2 = self.p**2
        if self.cfg.beta_spectrum:
            m2[BETA] *= _BETA23_M2
        return m2
