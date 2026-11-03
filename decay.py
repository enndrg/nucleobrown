from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GAMMA, KIND_INDEX, SourceConfig
from .isotopes import Isotope


@dataclass
class Events:
    times: np.ndarray  
    kinds: np.ndarray  
    stage: np.ndarray 
    primary: np.ndarray

    @staticmethod
    def empty() -> "Events":
        return Events(
            np.empty(0), np.empty(0, np.int8), np.empty(0, np.int16), np.empty(0, bool)
        )


def creation_times(source: SourceConfig, t_max: float, rng: np.random.Generator) -> np.ndarray:
    # creation noyau parents
    if source.mode == "batch":
        return np.zeros(source.n_nuclei)
    n = rng.poisson(source.rate * t_max)
    return rng.uniform(0.0, t_max, n)


def generate_events(
    chain: tuple[Isotope, ...],
    t_create: np.ndarray,
    t_max: float,
    rng: np.random.Generator,
) -> Events:
    t_cur = np.asarray(t_create, dtype=float)
    times, kinds, stages, prim = [], [], [], []
    for s, iso in enumerate(chain):
        if t_cur.size == 0:
            break
        t_dec = t_cur + rng.exponential(1.0 / iso.decay_rate, t_cur.size)
        t_dec = t_dec[t_dec <= t_max]
        n = t_dec.size
        if n == 0:
            break
        names, probs = iso.emission_probs()
        lut = np.array([KIND_INDEX[k] for k in names], dtype=np.int8)
        choice = rng.choice(len(names), size=n, p=probs)
        times.append(t_dec)
        kinds.append(lut[choice])
        stages.append(np.full(n, s, np.int16))
        prim.append(np.ones(n, bool))
        if iso.gamma_prob > 0:
            m = rng.random(n) < iso.gamma_prob
            k = int(m.sum())
            if k:
                times.append(t_dec[m])
                kinds.append(np.full(k, GAMMA, np.int8))
                stages.append(np.full(k, s, np.int16))
                prim.append(np.zeros(k, bool))
        t_cur = t_dec
    if not times:
        return Events.empty()
    t = np.concatenate(times)
    order = np.argsort(t, kind="stable")
    return Events(
        t[order],
        np.concatenate(kinds)[order],
        np.concatenate(stages)[order],
        np.concatenate(prim)[order],
    )
