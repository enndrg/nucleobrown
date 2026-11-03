#conversion JSON des objets numpy, sauvegarde/chargement des ensembles
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def to_jsonable(o):
    #(NaN/inf -> null)
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def save_ensemble(ens, path: str | Path) -> None:
    arrays = dict(grid=ens.grid, pos=ens.pos, counts=ens.counts, primary=ens.primary)
    if ens.vel is not None:
        arrays["vel"] = ens.vel
    np.savez_compressed(path, **arrays)


def load_ensemble(path: str | Path, cfg):
    from .simulation import Ensemble

    with np.load(path) as z:
        vel = z["vel"] if "vel" in z.files else None
        return Ensemble(
            cfg=cfg, grid=z["grid"], pos=z["pos"], vel=vel,
            counts=z["counts"], primary=z["primary"],
        )
