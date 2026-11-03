from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import Config
from .io_utils import save_ensemble, to_jsonable


def _load_cfg(args) -> Config:
    cfg = Config.load(args.config) if getattr(args, "config", None) else Config()
    cfg = cfg.with_overrides(args.set or [])
    if getattr(args, "workers", None):
        cfg = cfg.with_overrides([f"run.n_workers={args.workers}"])
    return cfg


def _outdir(args, prefix: str) -> Path:
    out = Path(args.out) if args.out else Path("results") / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def cmd_run(args) -> int:
    from .analysis import analyze
    from .simulation import simulate_ensemble

    cfg = _load_cfg(args)
    out = _outdir(args, "run")
    print(f"Simulation de {cfg.run.n_runs} trajectoires ({cfg.medium.regime}, dim {cfg.medium.dim})...", file=sys.stderr)
    t0 = time.time()
    ens = simulate_ensemble(cfg, verbose=True)
    summary, arrays = analyze(ens, cfg)
    cfg.save(out / "config.json")
    save_ensemble(ens, out / "ensemble.npz")
    with open(out / "report.json", "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(summary), fh, indent=2, ensure_ascii=False)
    if not args.no_plots:
        from .plotting import save_all

        save_all(ens, summary, arrays, cfg, out)
    fit = summary["msd_fit"]
    print(f"Terminé en {time.time() - t0:.1f} s -> {out}")
    print(f"  exposant alpha = {fit['alpha']:.3f}   D = {fit['D']:.4g}")
    if "diffusion_theory" in summary:
        d = summary["diffusion_theory"]
        print(f"  D théorique = {d['D_total']:.4g}  (écart {d['relative_error_fit']:+.1%})")
    print(f"  facteur de Fano = {summary['emissions']['fano_pooled']:.3f}")
    return 0


def cmd_sweep(args) -> int:
    from .analysis import analyze
    from .plotting import plot_sweep
    from .simulation import simulate_ensemble

    base = _load_cfg(args)
    out = _outdir(args, "sweep")
    values = [json.loads(v) for v in args.values.split(",")]
    rows = []
    for v in values:
        cfg = base.with_overrides([f"{args.param}={json.dumps(v)}"])
        summary, _ = analyze(simulate_ensemble(cfg), cfg)
        fit = summary["msd_fit"]
        d_th = summary.get("diffusion_theory", {}).get("D_total")
        rows.append(dict(value=v, D=fit["D"], D_std=fit.get("D_std", 0.0), D_theory=d_th, alpha=fit["alpha"]))
        print(f"  {args.param} = {v}: D = {fit['D']:.4g}, alpha = {fit['alpha']:.3f}", file=sys.stderr)
    with open(out / "sweep.json", "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(dict(param=args.param, rows=rows)), fh, indent=2)
    fig = plot_sweep(args.param, values, [r["D"] for r in rows], [r["D_std"] for r in rows],
                     [r["D_theory"] for r in rows], [r["alpha"] for r in rows])
    fig.savefig(out / "sweep.png", dpi=130)
    print(f"Balayage terminé -> {out}")
    return 0


def cmd_chains(_args) -> int:
    from .isotopes import CHAINS, describe_chain

    for name in CHAINS:
        print(describe_chain(name))
    return 0


def cmd_selftest(_args) -> int:
    from .selftest import run_selftests

    return 0 if run_selftests() else 1


def cmd_init(args) -> int:
    Config().save(args.path)
    print(f"Configuration par défaut écrite dans {args.path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nucleobrown", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", help="fichier de configuration JSON")
        sp.add_argument("--set", action="append", metavar="SECTION.CLE=VAL", help="surcharge (répétable)")
        sp.add_argument("--workers", type=int, help="processus parallèles")
        sp.add_argument("--out", help="dossier de sortie")

    r = sub.add_parser("run", help="simuler et analyser un ensemble de trajectoires")
    common(r)
    r.add_argument("--no-plots", action="store_true")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("sweep", help="balayer un paramètre")
    common(s)
    s.add_argument("--param", required=True, help="ex. medium.gamma")
    s.add_argument("--values", required=True, help="valeurs séparées par des virgules")
    s.set_defaults(fn=cmd_sweep)

    sub.add_parser("chains", help="lister les chaînes de désintégration").set_defaults(fn=cmd_chains)
    sub.add_parser("selftest", help="vérifier la simulation contre la théorie").set_defaults(fn=cmd_selftest)
    i = sub.add_parser("init-config", help="écrire une configuration par défaut")
    i.add_argument("path", nargs="?", default="config.json")
    i.set_defaults(fn=cmd_init)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except ValueError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2
