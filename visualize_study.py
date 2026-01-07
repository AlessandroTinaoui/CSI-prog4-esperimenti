from __future__ import annotations

import argparse
from pathlib import Path

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", default="sqlite:///xgboost_new.sqlite3")
    ap.add_argument("--study-name",default="xgboost_new_xgboostmodel")
    ap.add_argument("--outdir", default="optuna_plots")
    args = ap.parse_args()
    print("Studies in storage:")
    for s in optuna.get_all_study_summaries(storage=args.storage):
        print(" -", s.study_name)

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    figs = {
        "optimization_history.html": plot_optimization_history(study),
        "param_importances.html": plot_param_importances(study),
        "parallel_coordinate.html": plot_parallel_coordinate(study),
        "slice.html": plot_slice(study),
    }

    for name, fig in figs.items():
        path = outdir / name
        fig.write_html(str(path))
        print("Saved:", path)

    # opzionale: apre in browser automaticamente
    figs["optimization_history.html"].show()
    figs["parallel_coordinate.html"].show()
    figs["param_importances.html"].show()
    figs["slice.html"].show()


if __name__ == "__main__":
    main()
