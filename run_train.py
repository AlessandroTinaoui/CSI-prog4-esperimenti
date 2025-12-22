# run_train.py
from __future__ import annotations

import argparse
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

N = 1

@dataclass(frozen=True)
class ModelPaths:
    name: str
    server_dir: Path
    client_dir: Path

    @property
    def server_script(self) -> Path:
        return self.server_dir / "server_flwr.py"

    @property
    def client_script(self) -> Path:
        return self.client_dir / "run_all.py"

    @property
    def server_config(self) -> Path:
        return self.server_dir / "config.py"


MODELS = {
    "xgboost": ModelPaths(
        name="xgboost",
        server_dir=Path("xgboostmodel") / "server",
        client_dir=Path("xgboostmodel") / "client",
    ),
    "randomforest": ModelPaths(
        name="randomforest",
        server_dir=Path("randomforest") / "server",
        client_dir=Path("randomforest") / "client",
    ),
    "extratree": ModelPaths(
        name="extratree",
        server_dir=Path("extratreemodel") / "server",
        client_dir=Path("extratreemodel") / "client",
    ),
}

HOLDOUT_PATTERN = re.compile(r"^\s*HOLDOUT_CID\s*=\s*(.+?)\s*$", re.MULTILINE)
FINAL_MAE_PATTERN = re.compile(r"FINAL_MAE:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")


def ensure_paths(m: ModelPaths) -> None:
    for p in [m.server_dir, m.client_dir, m.server_script, m.client_script, m.server_config]:
        if not p.exists():
            raise FileNotFoundError(f"Path mancante: {p}")


def replace_holdout_cid(config_path: Path, cid: int) -> str:
    original = config_path.read_text(encoding="utf-8")
    if not HOLDOUT_PATTERN.search(original):
        raise ValueError(
            f"Non trovo 'HOLDOUT_CID = ...' in {config_path}. "
            "Aggiungi una riga tipo: HOLDOUT_CID = 0"
        )
    updated = HOLDOUT_PATTERN.sub(f"HOLDOUT_CID = {cid}", original)
    config_path.write_text(updated, encoding="utf-8")
    return original


def popen_logged(cmd: list[str], log_file: Path, cwd: Path | None = None) -> subprocess.Popen:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_file, "w", encoding="utf-8", buffering=1)  # line-buffered
    return subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def terminate_process(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def extract_final_mae_from_log(server_log: Path) -> float | None:
    if not server_log.exists():
        return None
    text = server_log.read_text(encoding="utf-8", errors="ignore")
    matches = FINAL_MAE_PATTERN.findall(text)
    if not matches:
        return None
    # prende l'ultimo, in caso di più stampe
    return float(matches[-1])


def run_one_training(
    m: ModelPaths,
    cid: int,
    rep: int,
    server_start_wait: float,
    logs_dir: Path,
    extra_server_args: list[str] | None = None,
    extra_client_args: list[str] | None = None,
) -> tuple[int, float | None, Path, Path]:
    extra_server_args = extra_server_args or []
    extra_client_args = extra_client_args or []

    server_log = logs_dir / f"{m.name}_cid{cid}_rep{rep}_server.log"
    client_log = logs_dir / f"{m.name}_cid{cid}_rep{rep}_client.log"

    py = sys.executable

    server_cmd = [py, "-u", "server_flwr.py"]
    server_proc = popen_logged(server_cmd, server_log, cwd=m.server_dir)

    time.sleep(server_start_wait)

    client_cmd = [py, "-u", "run_all.py"]
    client_proc = popen_logged(client_cmd, client_log, cwd=m.client_dir)

    client_rc = client_proc.wait()

    terminate_process(server_proc)

    mae = extract_final_mae_from_log(server_log)
    return client_rc, mae, server_log, client_log


def parse_cids(cids_str: str) -> list[int]:
    cids_str = cids_str.strip()
    if "-" in cids_str:
        a, b = cids_str.split("-", 1)
        start, end = int(a), int(b)
        return list(range(start, end + 1))
    return [int(x.strip()) for x in cids_str.split(",") if x.strip()]


def save_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # header stabile
    header = ["model", "holdout_cid", "repeat", "mae", "client_rc", "server_log", "client_log"]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r.get(k, "")) for k in header))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestratore training FL + summary MAE.")
    parser.add_argument("--model", choices=sorted(MODELS.keys()), default="extratree",
                        help="Modello da allenare (default: randomforest).")
    parser.add_argument("--repeats", type=int, default=N,
                        help="N: ripetizioni per ogni HOLDOUT_CID (default: 1).")
    parser.add_argument("--cids", type=str, default="0-8",
                        help="Range cids, es: '0-8' oppure '0,1,2,5'. (default: 0-8)")
    parser.add_argument("--server-wait", type=float, default=2.0,
                        help="Secondi di attesa dopo start server (default: 2.0).")
    parser.add_argument("--logs-dir", type=str, default="train_logs",
                        help="Cartella log (default: train_logs).")
    parser.add_argument("--server-args", type=str, default="",
                        help="Argomenti extra per server_flwr.py (stringa).")
    parser.add_argument("--client-args", type=str, default="",
                        help="Argomenti extra per run_all.py (stringa).")

    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats deve essere >= 1")

    m = MODELS[args.model]
    ensure_paths(m)

    logs_dir = Path(args.logs_dir)
    cids = parse_cids(args.cids)
    extra_server_args = args.server_args.split() if args.server_args.strip() else []
    extra_client_args = args.client_args.split() if args.client_args.strip() else []

    original_config = m.server_config.read_text(encoding="utf-8")

    def _handle_sigint(_sig, _frame):
        try:
            m.server_config.write_text(original_config, encoding="utf-8")
        finally:
            print("\nInterrotto. Config ripristinato.")
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _handle_sigint)

    runs: list[dict] = []

    try:
        for cid in cids:
            replace_holdout_cid(m.server_config, cid)

            for rep in range(1, args.repeats + 1):
                print(f"[{m.name}] HOLDOUT_CID={cid} | run {rep}/{args.repeats}")

                rc, mae, server_log, client_log = run_one_training(
                    m=m,
                    cid=cid,
                    rep=rep,
                    server_start_wait=args.server_wait,
                    logs_dir=logs_dir,
                    extra_server_args=extra_server_args,
                    extra_client_args=extra_client_args,
                )

                runs.append({
                    "model": m.name,
                    "holdout_cid": cid,
                    "repeat": rep,
                    "mae": "" if mae is None else mae,
                    "client_rc": rc,
                    "server_log": server_log.as_posix(),
                    "client_log": client_log.as_posix(),
                })

                if rc != 0:
                    print(f"⚠️ run_all.py exit code {rc} (vedi {client_log})")

                if mae is None:
                    print(f"⚠️ MAE non trovato nel server log {server_log}. "
                          f"Serve stampare 'FINAL_MAE: <valore>' dal server.")

        # ---- summary ----
        maes = [r["mae"] for r in runs if isinstance(r["mae"], (float, int))]
        if maes:
            print("\n=== SUMMARY MAE ===")
            print(f"MAE medio totale: {mean(maes):.6f}  (su {len(maes)} run con MAE trovato)")

            for cid in cids:
                cid_maes = [r["mae"] for r in runs
                            if r["holdout_cid"] == cid and isinstance(r["mae"], (float, int))]
                if cid_maes:
                    print(f"HOLDOUT_CID {cid}: MAE medio = {mean(cid_maes):.6f}  (n={len(cid_maes)})")
                else:
                    print(f"HOLDOUT_CID {cid}: MAE medio = N/A (nessun MAE trovato)")

        else:
            print("\nNessun MAE trovato nei log. "
                  "Consiglio di far stampare dal server una riga 'FINAL_MAE: <valore>'.")

        out_csv = logs_dir / f"mae_summary_{m.name}.csv"
        maes = [r["mae"] for r in runs if isinstance(r["mae"], (float, int))]
        mean_mae = None
        if maes:
            mean_mae = mean(maes)

            # Aggiungo una riga "finale" nel CSV con la media globale
            runs.append({
                "model": m.name,
                "holdout_cid": "ALL",
                "repeat": "",
                "mae": mean_mae,
                "client_rc": "",
                "server_log": "",
                "client_log": "",
            })

            # Scrivo anche un file testo con la media (comodo per parsing veloce)
            summary_txt = logs_dir / f"mae_summary_{m.name}.txt"
            summary_txt.write_text(f"MEAN_MAE: {mean_mae}\n", encoding="utf-8")

        save_csv(runs, out_csv)
        print(f"\nSalvato: {out_csv}")

        return 0

    finally:
        m.server_config.write_text(original_config, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
