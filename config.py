future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

KINDS = ("alpha", "beta", "gamma")
KIND_INDEX = {k: i for i, k in enumerate(KINDS)}
ALPHA, BETA, GAMMA = 0, 1, 2


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


@dataclass
class SourceConfig:
    """Source radioactive

    mode = "batch"      : noyaux « parents » créés à t = 0.
    mode = "continuous" : les noyaux parents sont créés par un processus de
                          Poisson, taux ``rate`` (source entretenue)
    """

    chain: str = "toy_chain"
    custom_chain: Optional[list] = None
    mode: str = "continuous"
    n_nuclei: int = 2000
    rate: float = 200.0

    def __post_init__(self) -> None:
        _check(self.mode in ("batch", "continuous"), "source.mode doit valoir 'batch' ou 'continuous'")
        _check(self.n_nuclei >= 0, "source.n_nuclei doit être >= 0")
        _check(self.rate > 0, "source.rate doit être > 0")


@dataclass
class KickConfig:
    # impulsions

    p_alpha: float = 1.0
    p_beta: float = 0.5
    p_gamma: float = 0.2
    beta_spectrum: bool = True  # spectre continu

    def __post_init__(self) -> None:
        for name in ("p_alpha", "p_beta", "p_gamma"):
            _check(getattr(self, name) >= 0, f"kicks.{name} doit être >= 0")


@dataclass
class MediumConfig:
# evol milieu

    dim: int = 2
    mass: float = 1.0
    gamma: float = 1.0
    kT: float = 0.0 
    regime: str = "overdamped" 

    def __post_init__(self) -> None:
        _check(self.dim in (1, 2, 3), "medium.dim doit valoir 1, 2 ou 3")
        _check(self.mass > 0, "medium.mass doit être > 0")
        _check(self.gamma >= 0, "medium.gamma doit être >= 0")
        _check(self.kT >= 0, "medium.kT doit être >= 0")
        _check(self.regime in ("overdamped", "underdamped"), "medium.regime invalide")
        if self.regime == "overdamped":
            _check(self.gamma > 0, "le régime 'overdamped' exige gamma > 0")
        if self.kT > 0:
            _check(self.gamma > 0, "kT > 0 exige gamma > 0 (relation fluctuation-dissipation)")


@dataclass
class RunConfig:
    t_max: float = 10.0
    n_grid: int = 200
    grid: str = "linear" 
    t_min: float = 1e-3
    n_runs: int = 200
    seed: int = 12345
    n_workers: int = 1

    def __post_init__(self) -> None:
        _check(self.t_max > 0, "run.t_max doit être > 0")
        _check(self.n_grid >= 4, "run.n_grid doit être >= 4")
        _check(self.grid in ("linear", "log"), "run.grid doit valoir 'linear' ou 'log'")
        _check(0 < self.t_min < self.t_max, "il faut 0 < run.t_min < run.t_max")
        _check(self.n_runs >= 2, "run.n_runs doit être >= 2")
        _check(self.n_workers >= 1, "run.n_workers doit être >= 1")


@dataclass
class AnalysisConfig:
    fit_window: list = field(default_factory=lambda: [0.3, 1.0])
    n_boot: int = 300
    snapshot_fracs: list = field(default_factory=lambda: [0.1, 0.4, 1.0])

    def __post_init__(self) -> None:
        _check(len(self.fit_window) == 2, "analysis.fit_window doit contenir 2 valeurs")
        lo, hi = self.fit_window
        _check(0 <= lo < hi <= 1, "analysis.fit_window doit vérifier 0 <= lo < hi <= 1")
        _check(self.n_boot >= 0, "analysis.n_boot doit être >= 0")
        _check(all(0 < f <= 1 for f in self.snapshot_fracs), "snapshot_fracs dans ]0, 1]")


_SECTIONS = {
    "source": SourceConfig,
    "kicks": KickConfig,
    "medium": MediumConfig,
    "run": RunConfig,
    "analysis": AnalysisConfig,
}


def _build(cls, data: Optional[dict]):
    data = dict(data or {})
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"Clés inconnues pour {cls.__name__} : {sorted(unknown)}")
    return cls(**data)


@dataclass
class Config:
    source: SourceConfig = field(default_factory=SourceConfig)
    kicks: KickConfig = field(default_factory=KickConfig)
    medium: MediumConfig = field(default_factory=MediumConfig)
    run: RunConfig = field(default_factory=RunConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Config":
        d = dict(d or {})
        unknown = set(d) - set(_SECTIONS)
        if unknown:
            raise ValueError(f"Sections inconnues : {sorted(unknown)}")
        return cls(**{name: _build(sub, d.get(name)) for name, sub in _SECTIONS.items()})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    def with_overrides(self, overrides: list[str]) -> "Config":
        #applique des surcharges
        d = self.to_dict()
        for item in overrides:
            key, sep, raw = item.partition("=")
            if not sep:
                raise ValueError(f"Surcharge invalide '{item}' (attendu section.cle=valeur)")
            section, dot, name = key.strip().partition(".")
            if not dot or section not in d or name not in d[section]:
                raise ValueError(f"Paramètre inconnu : '{key}'")
            try:
                value: Any = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            d[section][name] = value
        return Config.from_dict(d)
