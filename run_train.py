# run_train.py
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import json
import hashlib
from datetime import datetime

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
    # mantengo "xgboost" se ti serve ancora da CLI legacy
    "xgboost": ModelPaths(
        name="xgboost",
        server_dir=Path("xgboostmodel") / "server",
        client_dir=Path("xgboostmodel") / "client",
    ),
    # alias coerente con Optuna/search_space/study_driver
    "xgboostmodel": ModelPaths(
        name="xgboostmodel",
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
    "mlp": ModelPaths(
        name="mlp",
        server_dir=Path("mlp") / "server",
        client_dir=Path("mlp") / "client",
    ),
    "tabnet": ModelPaths(
        name="TabNet",
        server_dir=Path("TabNet") / "server",
        client_dir=Path("TabNet") / "client",
    ),
}

HOLDOUT_PATTERN = re.compile(r"^\s*HOLDOUT_CID\s*=\s*(.+?)\s*$", re.MULTILINE)
FINAL_MAE_PATTERN = re.compile(r"FINAL_MAE:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")

def log_trial_config_debug(logs_dir: Path, model_name: str, cid: int, rep: int) -> None:
    """
    Scrive su file:
      - TRIAL_CONFIG_PATH visto dal processo
      - esistenza file
      - sha256 contenuto
      - primi 200 char
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    out = logs_dir / "trial_config_debug.txt"

    cfg_path = os.environ.get("TRIAL_CONFIG_PATH", "")
    ts = datetime.now().isoformat(timespec="seconds")

    lines = []
    lines.append(f"[{ts}] model={model_name} cid={cid} rep={rep}")
    lines.append(f"TRIAL_CONFIG_PATH={cfg_path!r}")

    if not cfg_path:
        lines.append("STATUS: MISSING_ENV")
        lines.append("-" * 80)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8") if not out.exists() else out.open("a", encoding="utf-8").write("\n".join(lines) + "\n")
        return

    p = Path(cfg_path)
    lines.append(f"EXISTS={p.exists()}  ABS={p.resolve() if p.exists() else p}")

    if p.exists():
        content = p.read_text(encoding="utf-8", errors="replace")
        sha = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
        preview = content[:200].replace("\n", "\\n")
        lines.append(f"SHA256={sha}")
        lines.append(f"PREVIEW_200={preview!r}")
    else:
        lines.append("STATUS: PATH_NOT_FOUND")

    lines.append("-" * 80)

    # append safe
    with out.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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

    log_trial_config_debug(logs_dir=logs_dir, model_name=m.name, cid=cid, rep=rep)

    py = sys.executable

    BASE_PORT = 20000
    PORT_OFFSET = 50
    port = BASE_PORT + cid + rep * PORT_OFFSET
    server_address = f"127.0.0.1:{port}"

    env = os.environ.copy()
    env["HOLDOUT_CID"] = str(cid)
    env["FL_SERVER_ADDRESS"] = server_address
    # TRIAL_CONFIG_PATH arriva da env esterno (Optuna/study_driver): lo preserviamo
    # RUN_DIR arriva da env esterno: lo preserviamo

    server_cmd = [py, "-u", "server_flwr.py"] + extra_server_args
    client_cmd = [py, "-u", "run_all.py"] + extra_client_args

    server_log.parent.mkdir(parents=True, exist_ok=True)
    client_log.parent.mkdir(parents=True, exist_ok=True)

    sf = open(server_log, "w", encoding="utf-8", buffering=1)
    cf = open(client_log, "w", encoding="utf-8", buffering=1)

    server_proc = None
    client_proc = None
    try:
        server_proc = subprocess.Popen(
            server_cmd,
            cwd=str(m.server_dir),
            env=env,
            stdout=sf,
            stderr=subprocess.STDOUT,
            text=True,
        )

        time.sleep(server_start_wait)

        client_proc = subprocess.Popen(
            client_cmd,
            cwd=str(m.client_dir),
            env=env,
            stdout=cf,
            stderr=subprocess.STDOUT,
            text=True,
        )

        client_rc = client_proc.wait()
        time.sleep(0.5)

    finally:
        try:
            if server_proc is not None:
                terminate_process(server_proc)
        finally:
            sf.close()
            cf.close()

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
    header = ["model", "holdout_cid", "repeat", "mae", "client_rc", "server_log", "client_log"]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r.get(k, "")) for k in header))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestratore training FL + summary MAE.")
    parser.add_argument(
        "--model",
        choices=sorted(MODELS.keys()),
        default="tabnet",
        help="Modello da allenare.",
    )
    parser.add_argument("--repeats", type=int, default=N, help="Ripetizioni per ogni HOLDOUT_CID (default: 1).")
    parser.add_argument("--cids", type=str, default="0-8", help="Range cids, es: '0-8' oppure '0,1,2,5'.")
    parser.add_argument("--server-wait", type=float, default=2.0, help="Secondi di attesa dopo start server.")
    parser.add_argument("--logs-dir", type=str, default="train_logs", help="Cartella log / output.")
    parser.add_argument("--server-args", type=str, default="", help="Argomenti extra per server_flwr.py (stringa).")
    parser.add_argument("--client-args", type=str, default="", help="Argomenti extra per run_all.py (stringa).")

    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats deve essere >= 1")

    m = MODELS[args.model]
    ensure_paths(m)

    logs_dir = Path(args.logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

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
            # Nota: questo modifica config.py. Va bene in sequenziale (Optuna n_jobs=1).
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
                    "model": args.model,  # importante: coerente col nome richiesto in output
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
            mean_mae = mean(maes)

            out_csv = logs_dir / f"mae_summary_{args.model}.csv"
            runs.append({
                "model": args.model,
                "holdout_cid": "ALL",
                "repeat": "",
                "mae": mean_mae,
                "client_rc": "",
                "server_log": "",
                "client_log": "",
            })
            save_csv(runs, out_csv)

            summary_txt = logs_dir / f"mae_summary_{args.model}.txt"
            summary_txt.write_text(f"MEAN_MAE: {mean_mae}\n", encoding="utf-8")

            print("\n=== SUMMARY MAE ===")
            print(f"MEAN_MAE: {mean_mae:.6f}  (su {len(maes)} run con MAE trovato)")
            print(f"Salvati: {out_csv} e {summary_txt}")

            return 0

        # Se non trovi MAE, non scrivo summary: Optuna deve fallire il trial
        out_csv = logs_dir / f"mae_summary_{args.model}.csv"
        save_csv(runs, out_csv)
        print("\nNessun MAE trovato nei log. Trial fallirà (manca mae_summary txt).")
        print(f"Salvato comunque il CSV: {out_csv}")
        return 2

    finally:
        m.server_config.write_text(original_config, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
