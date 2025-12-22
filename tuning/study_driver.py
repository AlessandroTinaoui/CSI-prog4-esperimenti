# tuning/study_driver.py
# Optuna driver con selezione automatica degli script in base al modello

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import optuna

from tuning.search_space import suggest_params
from tuning.trial_io import save_json, load_json


# =========================
# MAPPING MODELLI → SCRIPT
# =========================
MODEL_REGISTRY = {
    "xgboostmodel": {
        "server": "xgboostmodel/server/server_flwr.py",
        "client": "xgboostmodel/client/run_all.py",
    },
    "randomforest": {
        "server": "randomforest/server/server_flwr.py",
        "client": "randomforest/client/run_all.py",
    },
    # in futuro:
    # "decisiontree": {...}
}


def main() -> None:
    ap = argparse.ArgumentParser()

    # 🔧 modello
    ap.add_argument(
        "--model",
        default="xgboostmodel",
        choices=MODEL_REGISTRY.keys(),
        help="Modello da usare (sceglie automaticamente server/client)",

    )

    # Optuna
    ap.add_argument("--study-name", default="fl_tree_tuning")
    ap.add_argument("--storage", default="sqlite:///optuna_study.db")
    ap.add_argument("--n-trials", type=int, default=30)

    # Benchmark (holdout uno alla volta)
    ap.add_argument("--holdout-range", default="0-8")
    ap.add_argument("--repeats", type=int, default=1)

    # output
    ap.add_argument("--runs-dir", default="runs")

    args = ap.parse_args()

    # =========================
    # PATH DI BASE
    # =========================
    tuning_dir = Path(__file__).resolve().parent
    project_root = tuning_dir.parent

    runs_root = (project_root / args.runs_dir).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    # =========================
    # BENCHMARK SCRIPT
    # =========================
    benchmark_script = (tuning_dir / "run_holdout_benchmark.py").resolve()
    if not benchmark_script.exists():
        raise FileNotFoundError(f"Benchmark script non trovato: {benchmark_script}")

    # =========================
    # SCRIPT SERVER / CLIENT
    # =========================
    model_entry = MODEL_REGISTRY[args.model]

    server_script = (project_root / model_entry["server"]).resolve()
    client_script = (project_root / model_entry["client"]).resolve()

    if not server_script.exists():
        raise FileNotFoundError(f"Server script non trovato: {server_script}")
    if not client_script.exists():
        raise FileNotFoundError(f"Client script non trovato: {client_script}")

    print(f"[INFO] Modello selezionato: {args.model}")
    print(f"[INFO] Server: {server_script}")
    print(f"[INFO] Client: {client_script}")

    # =========================
    # STUDY OPTUNA
    # =========================
    study = optuna.create_study(
        study_name=f"{args.study_name}_{args.model}",
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
    )

    # =========================
    # OBJECTIVE
    # =========================
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)

        trial_dir = runs_root / f"{args.study_name}_{args.model}" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        trial_config_path = trial_dir / "trial_config.json"
        save_json(trial_config_path, params)

        out_dir = trial_dir / "benchmark"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-u",
            str(benchmark_script),
            "--trial-config",
            str(trial_config_path),
            "--out-dir",
            str(out_dir),
            "--holdout-range",
            args.holdout_range,
            "--repeats",
            str(args.repeats),
            "--server-script",
            str(server_script),
            "--client-script",
            str(client_script),
            "--workdir",
            str(project_root),
        ]

        p = subprocess.run(cmd, text=True, capture_output=True, cwd=str(project_root))

        (out_dir / "benchmark_stdout.txt").write_text(p.stdout)
        (out_dir / "benchmark_stderr.txt").write_text(p.stderr)

        if p.returncode != 0:
            raise RuntimeError(
                f"\n❌ Benchmark failed (trial {trial.number})\n"
                f"CMD: {' '.join(cmd)}\n"
                f"--- STDERR ---\n{p.stderr}\n"
                f"--- STDOUT ---\n{p.stdout}\n"
            )

        result = load_json(out_dir / "result.json")
        return float(result["mean_mae"])

    # =========================
    # RUN OPTUNA
    # =========================
    study.optimize(objective, n_trials=args.n_trials, catch=(Exception,))

    best = study.best_trial
    print("\n=== BEST TRIAL ===")
    print("value (mean_mae):", best.value)
    print("params:", best.params)


if __name__ == "__main__":
    main()
