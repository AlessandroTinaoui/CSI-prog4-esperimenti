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
    ap.add_argument("--storage", default="sqlite:///optuna_study.sqlite3")
    ap.add_argument("--study-name", required=True)
    ap.add_argument("--outdir", default="optuna_plots")
    args = ap.parse_args()

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
    figs["parallel_coordinate.html.html"].show()
    figs["param_importances.html.html"].show()
    figs["slice.html.html"].show()


if __name__ == "__main__":
    main()
