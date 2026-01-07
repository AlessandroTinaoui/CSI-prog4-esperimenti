# tuning/study_driver.py
# Optuna driver: 1 trial = 1 chiamata a run_train.py
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import optuna

from tuning.search_space import suggest_params
from tuning.trial_io import save_json


# =========================
# MAPPING MODELLI → SCRIPT (solo per choices/consistenza)
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
    "extratree": {
        "server": "extratreemodel/server/server_flwr.py",
        "client": "extratreemodel/client/run_all.py",
    },
    "mlp": {
        "server": "mlp/server/server_flwr.py",
        "client": "mlp/client/run_all.py",
    },
    "TabNet": {
        "server": "TabNet/server/server_flwr.py",
        "client": "TabNet/client/run_all.py",
    },
}


def parse_mean_mae(summary_path: Path) -> float:
    if not summary_path.exists():
        raise FileNotFoundError(f"File summary non trovato: {summary_path}")
    line = summary_path.read_text(encoding="utf-8").strip()
    if not line.startswith("MEAN_MAE:"):
        raise ValueError(f"Formato inatteso in {summary_path}: {line!r}")
    return float(line.split(":", 1)[1].strip())


def main() -> None:
    ap = argparse.ArgumentParser()

    # modello
    ap.add_argument(
        "--model",
        default="xgboostmodel",
        choices=MODEL_REGISTRY.keys(),
        help="Modello da usare",
    )

    # Optuna
    ap.add_argument("--study-name", default="mlp_prepro2_3")
    ap.add_argument("--storage", default="sqlite:///Mlp_prepro2_3.sqlite3")
    ap.add_argument("--n-trials", type=int, default=100)

    # training runner
    ap.add_argument("--cids", type=str, default="0-8", help="Range cids, es: '0-8' o '0,1,2,5'")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--server-wait", type=float, default=2.0)
    ap.add_argument("--server-args", type=str, default="", help="Argomenti extra per server_flwr.py (stringa)")
    ap.add_argument("--client-args", type=str, default="", help="Argomenti extra per run_all.py (stringa)")

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

    run_train_script = (project_root / "run_train.py").resolve()
    if not run_train_script.exists():
        raise FileNotFoundError(f"run_train.py non trovato: {run_train_script}")

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
        # serve per scegliere lo spazio di ricerca corretto
        trial.set_user_attr("model", args.model)
        params = suggest_params(trial)

        # cartella trial
        trial_dir = runs_root / f"{args.study_name}_{args.model}" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # salva config del trial
        trial_config_path = trial_dir / "trial_config.json"
        save_json(trial_config_path, params)

        # chiama run_train e digli di scrivere dentro trial_dir
        cmd = [
            sys.executable,
            "-u",
            str(run_train_script),
            "--model",
            args.model,
            "--repeats",
            str(args.repeats),
            "--cids",
            args.cids,
            "--server-wait",
            str(args.server_wait),
            "--logs-dir",
            str(trial_dir),
        ]
        if args.server_args.strip():
            cmd += ["--server-args", args.server_args]
        if args.client_args.strip():
            cmd += ["--client-args", args.client_args]

        # env: fondamentale per far leggere gli iperparametri ai client
        env = os.environ.copy()
        env["TRIAL_CONFIG_PATH"] = str(trial_config_path)
        env["RUN_DIR"] = str(trial_dir)  # utile se in futuro vuoi far rispettare RUN_DIR anche ai client

        driver_log = trial_dir / "run_train_driver.log"
        p = subprocess.run(
            cmd,
            text=True,
            cwd=str(project_root),
            env=env,
            stdout=driver_log.open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
        if p.returncode != 0:
            raise RuntimeError(f"run_train fallito (trial {trial.number}). Vedi {driver_log}")

        # leggi MEAN_MAE dalla summary salvata nel trial_dir
        summary_txt = trial_dir / f"mae_summary_{args.model}.txt"
        return parse_mean_mae(summary_txt)

    # =========================
    # RUN OPTUNA
    # =========================
    study.optimize(objective, n_trials=args.n_trials, catch=(Exception,))

    best = study.best_trial
    print("\n=== BEST TRIAL ===")
    print("value (MEAN_MAE):", best.value)
    print("params:", best.params)


if __name__ == "__main__":
    main()
