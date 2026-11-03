from __future__ import annotations

import functools
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Config, RunConfig
from .decay import creation_times, generate_events
from .dynamics import propagate
from .isotopes import resolve_chain
from .kicks import KickModel


def make_grid(run: RunConfig) -> np.ndarray:
    if run.grid == "linear":
        return np.linspace(0.0, run.t_max, run.n_grid + 1)
    return np.concatenate([[0.0], np.geomspace(run.t_min, run.t_max, run.n_grid)])


@dataclass
class RunResult:
    pos: np.ndarray  # (G+1, dim)
    vel: Optional[np.ndarray]  # (G+1, dim) ou None
    counts: np.ndarray  # (G+1, 3) émissions cumulées
    primary: np.ndarray  # (G+1,) émissions 1
    events: Optional[dict] = None 


def simulate_run(cfg: Config, seed, full: bool = False) -> RunResult:
    """Unique réal. Les points de la grille d'échantillonnage sont insérés comme
    des chocs nuls. La propagation étant exacte et markovienne ->
    exactement la loi de la trajectoire échantillonnée."""
    rng = np.random.default_rng(seed)
    chain = resolve_chain(cfg.source)
    dim = cfg.medium.dim

    t_create = creation_times(cfg.source, cfg.run.t_max, rng)
    ev = generate_events(chain, t_create, cfg.run.t_max, rng)
    impulses = KickModel(cfg.kicks, dim).sample(ev.kinds, rng)

    grid = make_grid(cfg.run)
    G, K = grid.size - 1, ev.times.size
    times_all = np.concatenate([ev.times, grid[1:]])
    imp_all = np.concatenate([impulses, np.zeros((G, dim))], axis=0)
    order = np.argsort(times_all, kind="stable")
    t_sorted = times_all[order]

    x, v, v0 = propagate(t_sorted, imp_all[order], cfg.medium, rng)

    inv = np.empty(order.size, dtype=np.int64)
    inv[order] = np.arange(order.size)
    gi = inv[K:]
    pos = np.vstack([np.zeros((1, dim)), x[gi]])
    vel = None if v is None else np.vstack([v0[None, :], v[gi]])

    counts = np.stack(
        [np.searchsorted(ev.times[ev.kinds == k], grid, side="right") for k in range(3)], axis=1
    )
    primary = np.searchsorted(ev.times[ev.primary], grid, side="right")

    events = None
    if full:
        is_ev = (order < K)
        events = dict(
            times=t_sorted[is_ev],
            kinds=ev.kinds[order[is_ev]],
            pos=x[is_ev],
            vel=None if v is None else v[is_ev],
        )
    return RunResult(pos, vel, counts.astype(np.int32), primary.astype(np.int32), events)


@dataclass
class Ensemble:
    cfg: Config
    grid: np.ndarray  # (G+1,)
    pos: np.ndarray  # (M, G+1, dim)
    vel: Optional[np.ndarray]
    counts: np.ndarray  # (M, G+1, 3)
    primary: np.ndarray  # (M, G+1)
    showcase: Optional[dict] = None  

def _worker(cfg: Config, seed) -> RunResult:
    return simulate_run(cfg, seed)


def simulate_ensemble(cfg: Config, verbose: bool = False) -> Ensemble:
    M = cfg.run.n_runs
    seeds = np.random.SeedSequence(cfg.run.seed).spawn(M + 1)  #traj finale
    workers = cfg.run.n_workers
    if workers <= 1:
        results = []
        for i, s in enumerate(seeds[:M]):
            results.append(simulate_run(cfg, s))
            if verbose and (i + 1) % max(1, M // 10) == 0:
                print(f"  trajectoires : {i + 1}/{M}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            chunk = max(1, M // (4 * workers))
            results = list(ex.map(functools.partial(_worker, cfg), seeds[:M], chunksize=chunk))

    show = simulate_run(cfg, seeds[M], full=True).events
    vel = None if results[0].vel is None else np.stack([r.vel for r in results])
    return Ensemble(
        cfg=cfg,
        grid=make_grid(cfg.run),
        pos=np.stack([r.pos for r in results]),
        vel=vel,
        counts=np.stack([r.counts for r in results]),
        primary=np.stack([r.primary for r in results]),
        showcase=show,
    )
