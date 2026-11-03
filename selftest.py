from __future__ import annotations

from .analysis import analyze, squared_displacement
from .config import Config
from .simulation import simulate_ensemble


def _run(d: dict):
    cfg = Config.from_dict(d)
    ens = simulate_ensemble(cfg)
    summary, arrays = analyze(ens, cfg)
    return cfg, ens, summary, arrays


def check_poisson_single_stage():
    *_, s, _ = _run(dict(
        source=dict(chain="toy_single", mode="continuous", rate=300),
        run=dict(n_runs=300, t_max=10, n_grid=100, seed=1)))
    f = s["emissions"]["fano_pooled"]
    return abs(f - 1.0) < 0.05, f"Fano groupé = {f:.3f} (attendu 1)"


def check_chain_correlations():
    *_, s, _ = _run(dict(
        source=dict(chain="toy_fast", mode="continuous", rate=300),
        run=dict(n_runs=300, t_max=10, n_grid=100, seed=2)))
    f = s["emissions"]["fano_pooled"]
    return f > 1.05, f"Fano groupé = {f:.3f} (attendu > 1)"


def check_batch_subpoisson():
    *_, s, _ = _run(dict(
        source=dict(chain="toy_chain", mode="batch", n_nuclei=2000),
        run=dict(n_runs=300, t_max=10, n_grid=20, seed=3)))
    f = s["emissions"]["fano_pooled"]
    return f < 0.95, f"Fano groupé = {f:.3f} (attendu < 1)"


def check_msd_overdamped_batch():
    *_, s, _ = _run(dict(
        source=dict(chain="toy_chain", mode="batch", n_nuclei=1500),
        medium=dict(regime="overdamped", gamma=1.5, mass=1.0, dim=2),
        run=dict(n_runs=300, t_max=12, seed=4)))
    v = s["msd_vs_theory"]
    return v["max_abs_z"] < 4.5 and abs(v["mean_z"]) < 1.0, (
        f"max|z| = {v['max_abs_z']:.2f}, <z> = {v['mean_z']:.2f}")


def check_diffusion_continuous():
    *_, s, _ = _run(dict(
        source=dict(chain="toy_fast", mode="continuous", rate=300),
        run=dict(n_runs=400, t_max=10, seed=105)))
    d = s["diffusion_theory"]
    z = d.get("z_fit", float("nan"))
    return abs(z) < 3, f"erreur relative = {d['relative_error_fit']:+.1%}, z = {z:+.2f}"


def check_thermal_overdamped():
    *_, s, _ = _run(dict(
        source=dict(mode="batch", n_nuclei=0),
        medium=dict(regime="overdamped", gamma=2.0, kT=0.8, dim=3),
        run=dict(n_runs=400, t_max=5, seed=6)))
    d = s["diffusion_theory"] if "diffusion_theory" in s else None
    D, sd = s["msd_fit"]["D"], s["msd_fit"]["D_std"]
    D_th = 0.8 / (1.0 * 2.0)
    z = (D - D_th) / sd
    return abs(z) < 3, f"D = {D:.3f} ± {sd:.3f}, attendu {D_th:.3f} (z = {z:+.2f})"


def check_underdamped_diffusion():
    *_, s, _ = _run(dict(
        source=dict(chain="toy_fast", mode="continuous", rate=300),
        medium=dict(regime="underdamped", gamma=2.0, kT=0.5, dim=2),
        run=dict(n_runs=300, t_max=20, seed=7)))
    d = s["diffusion_theory"]
    z = d.get("z_fit", float("nan"))
    return abs(z) < 3, f"erreur relative = {d['relative_error_fit']:+.1%}, z = {z:+.2f}"


def check_ou_exactness():
    import numpy as np

    cfg, ens, s, a = _run(dict(
        source=dict(mode="batch", n_nuclei=0),
        medium=dict(regime="underdamped", gamma=1.0, kT=2.0, mass=1.0, dim=2),
        run=dict(n_runs=600, t_max=6, n_grid=120, seed=8)))
    t = a["t"]
    th = 2 * 2 * 2.0 / 1.0**2 * (t - 1 + np.exp(-t))
    r2 = squared_displacement(ens.pos)
    sem = r2.std(axis=0, ddof=1) / np.sqrt(r2.shape[0])
    z = (r2.mean(0)[1:] - th[1:]) / sem[1:]
    v2 = float(np.mean(ens.vel**2))
    ok = np.abs(z).max() < 4.5 and abs(v2 - 2.0) / 2.0 < 0.03
    return bool(ok), f"max|z| MSD = {np.abs(z).max():.2f}, <v_i^2> = {v2:.3f} (attendu 2)"


def check_ballistic_exponent():
    *_, s, _ = _run(dict(
        source=dict(chain="toy_fast", mode="continuous", rate=200),
        medium=dict(regime="underdamped", gamma=0.0, kT=0.0, dim=2),
        run=dict(n_runs=300, t_max=10, seed=9),
        analysis=dict(fit_window=[0.3, 1.0])))
    al = s["msd_fit"]["alpha"]
    return abs(al - 3.0) < 0.25, f"alpha = {al:.2f} (attendu 3)"


ALL_CHECKS = [
    check_poisson_single_stage,
    check_chain_correlations,
    check_batch_subpoisson,
    check_msd_overdamped_batch,
    check_diffusion_continuous,
    check_thermal_overdamped,
    check_underdamped_diffusion,
    check_ou_exactness,
    check_ballistic_exponent,
]


def run_selftests(verbose: bool = True) -> bool:
    all_ok = True
    for fn in ALL_CHECKS:
        try:
            ok, msg = fn()
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, f"exception : {exc!r}"
        all_ok &= ok
        if verbose:
            print(f"[{'OK ' if ok else 'ÉCHEC'}] {fn.__name__:<32} {msg}")
    return all_ok
