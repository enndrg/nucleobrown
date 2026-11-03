from __future__ import annotations

import math
from dataclasses import dataclass

from .config import KIND_INDEX, SourceConfig

LN2 = math.log(2.0)


@dataclass(frozen=True)
class Isotope:
    name: str
    half_life: float  #sec
    emissions: tuple = (("alpha", 1.0),)  #distrib du rayonnement parent
    gamma_prob: float = 0.0  # probab d'un photon gamma accompagnant la désintégration

    def __post_init__(self) -> None:
        if not (self.half_life > 0 and math.isfinite(self.half_life)):
            raise ValueError(f"{self.name} : demi-vie finie et > 0 requise")
        if not self.emissions:
            raise ValueError(f"{self.name} : au moins une émission primaire requise")
        total = 0.0
        for kind, p in self.emissions:
            if kind not in KIND_INDEX:
                raise ValueError(f"{self.name} : type d'émission inconnu '{kind}'")
            if p < 0:
                raise ValueError(f"{self.name} : probabilité négative")
            total += p
        if total <= 0:
            raise ValueError(f"{self.name} : somme des probabilités nulle")
        if not 0 <= self.gamma_prob <= 1:
            raise ValueError(f"{self.name} : gamma_prob doit être dans [0, 1]")

    @property
    def decay_rate(self) -> float:
        return LN2 / self.half_life

    @property
    def mean_life(self) -> float:
        return 1.0 / self.decay_rate

    def emission_probs(self) -> tuple[list[str], list[float]]:
        names = [k for k, _ in self.emissions]
        probs = [p for _, p in self.emissions]
        s = sum(probs)
        return names, [p / s for p in probs]


def chain_from_spec(spec: list[dict]) -> tuple[Isotope, ...]:
    #depuis json
    chain = []
    for item in spec:
        emissions = item.get("emissions", {"alpha": 1.0})
        chain.append(
            Isotope(
                name=item["name"],
                half_life=float(item["half_life"]),
                emissions=tuple((k, float(v)) for k, v in emissions.items()),
                gamma_prob=float(item.get("gamma_prob", 0.0)),
            )
        )
    if not chain:
        raise ValueError("chaîne vide")
    return tuple(chain)


CHAINS: dict[str, tuple[Isotope, ...]] = {
    "toy_single": (Isotope("X", 2.0, (("alpha", 1.0),)),),

    "toy_chain": (
        Isotope("A", 1.0, (("alpha", 1.0),)),
        Isotope("B", 3.0, (("beta", 1.0),), gamma_prob=0.5),
    ),

    "toy_fast": (
        Isotope("A", 0.05, (("alpha", 1.0),)),
        Isotope("B", 0.10, (("beta", 1.0),), gamma_prob=0.5),
    ),

    "toy_stiff": (
        Isotope("S1", 0.01, (("alpha", 1.0),)),
        Isotope("S2", 5.0, (("beta", 1.0),), gamma_prob=0.3),
        Isotope("S3", 200.0, (("alpha", 1.0),)),
    ),
    #du radon-222 (ordres de grandeur, sec)
    "rn222": (
        Isotope("Rn-222", 3.8235 * 86400.0, (("alpha", 1.0),), gamma_prob=0.001),
        Isotope("Po-218", 185.9, (("alpha", 1.0),)),
        Isotope("Pb-214", 1608.0, (("beta", 1.0),), gamma_prob=0.5),
        Isotope("Bi-214", 1194.0, (("beta", 0.9998), ("alpha", 0.0002)), gamma_prob=0.9),
        Isotope("Po-214", 164.3e-6, (("alpha", 1.0),)),
    ),
}


def get_chain(name: str) -> tuple[Isotope, ...]:
    try:
        return CHAINS[name]
    except KeyError:
        raise ValueError(f"Chaîne inconnue '{name}'. Disponibles : {sorted(CHAINS)}") from None


def resolve_chain(source: SourceConfig) -> tuple[Isotope, ...]:
    if source.custom_chain:
        return chain_from_spec(source.custom_chain)
    return get_chain(source.chain)


def describe_chain(name: str) -> str:
    lines = [f"{name}:"]
    for s, iso in enumerate(get_chain(name)):
        em = ", ".join(f"{k} {p:g}" for k, p in iso.emissions)
        lines.append(
            f"  [{s}] {iso.name:<8} T1/2 = {iso.half_life:<12.6g} s  émissions : {em}"
            + (f"  (+ gamma {iso.gamma_prob:g})" if iso.gamma_prob else "")
        )
    return "\n".join(lines)
