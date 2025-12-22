# run_holdout_benchmark.py

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

BASE_PORT = 18080
PORT_OFFSET = 20


FINAL_MAE_RE = re.compile(r"FINAL_MAE:\s*([0-9]*\.?[0-9]+)")


def parse_final_mae(log_text: str) -> float:
    m = FINAL_MAE_RE.search(log_text)
    if not m:
        raise RuntimeError("Impossibile trovare 'FINAL_MAE: <val>' nel log server.")
    return float(m.group(1))


def load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def save_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


def holdout_list_from_range(s: str) -> list[int]:
    # accetta "0-8" oppure "0,1,2"
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run_one(server_py: str, client_py: str, env: dict, cwd: Path, server_log: Path, client_log: Path) -> str:
    """
    Avvia server e poi client. Ritorna il testo del log server (per parsing).
    """
    server_log.parent.mkdir(parents=True, exist_ok=True)
    client_log.parent.mkdir(parents=True, exist_ok=True)

    # Server
    with server_log.open("w") as sl:
        ps = subprocess.Popen(
            [sys.executable, "-u", server_py],
            cwd=str(cwd),
            env=env,
            stdout=sl,
            stderr=subprocess.STDOUT,
            text=True,
        )

    # Aspetta che il server salga un minimo
    time.sleep(2.0)

    # Clients
    with client_log.open("w") as cl:
        pc = subprocess.Popen(
            [sys.executable, "-u", client_py],
            cwd=str(cwd),
            env=env,
            stdout=cl,
            stderr=subprocess.STDOUT,
            text=True,
        )

    # Attendi client
    rc_client = pc.wait()
    time.sleep(0.5)
    # Attendi server
    rc_server = ps.wait()

    # Leggi log server
    text = server_log.read_text(errors="ignore")

    if rc_client != 0:
        raise RuntimeError(f"Client process terminato con codice {rc_client}. Vedi: {client_log}")
    if rc_server != 0:
        raise RuntimeError(f"Server process terminato con codice {rc_server}. Vedi: {server_log}")

    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-config", required=True, help="Path al trial_config.json")
    ap.add_argument("--out-dir", required=True, help="Directory output (logs + result.json)")
    ap.add_argument("--holdout-range", default="0-8", help='Es: "0-8" oppure "0,1,2"')
    ap.add_argument("--repeats", type=int, default=1)

    # Dove sono gli script server/client (adatta se necessario)
    ap.add_argument("--server-script", default="server_flwr.py")
    ap.add_argument("--client-script", default="run_all.py")

    # Working dir da cui lanciare i processi (di solito la cartella che contiene server_flwr.py)
    ap.add_argument("--workdir", default=".", help="Directory di lavoro per lanciare server/client")
    args = ap.parse_args()

    trial_cfg_path = Path(args.trial_config).resolve()
    out_dir = Path(args.out_dir).resolve()
    workdir = Path(args.workdir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_json(trial_cfg_path)
    holdouts = holdout_list_from_range(args.holdout_range)

    mae_by_holdout: dict[str, float] = {}
    failures: dict[str, str] = {}

    for rep in range(args.repeats):
        for cid in holdouts:
            port = BASE_PORT + cid + rep * PORT_OFFSET
            server_address = f"127.0.0.1:{port}"

            env = os.environ.copy()

            # Config runtime per questo holdout (non tocchi il base trial_config)
            runtime_cfg = json.loads(json.dumps(base_cfg))
            runtime_cfg.setdefault("server", {})
            runtime_cfg["server"]["HOLDOUT_CID"] = int(cid)

            run_dir = out_dir / f"rep_{rep}" / f"holdout_{cid}"
            logs_dir = run_dir / "logs"
            cfg_dir = run_dir / "configs"
            runtime_cfg_path = cfg_dir / "runtime_config.json"
            save_json(runtime_cfg_path, runtime_cfg)

            env = os.environ.copy()
            env["TRIAL_CONFIG_PATH"] = str(runtime_cfg_path)
            env["HOLDOUT_CID"] = str(cid)
            env["FL_SERVER_ADDRESS"] = server_address

            env["RUN_DIR"] = str(run_dir)

            server_log = logs_dir / "server.log"
            client_log = logs_dir / "client.log"

            try:
                server_text = run_one(
                    server_py=args.server_script,
                    client_py=args.client_script,
                    env=env,
                    cwd=workdir,
                    server_log=server_log,
                    client_log=client_log,
                )
                mae = parse_final_mae(server_text)
                # se repeats=1, la chiave è solo il cid; se repeats>1, incorporiamo anche rep
                key = f"{cid}" if args.repeats == 1 else f"{cid}_rep{rep}"
                mae_by_holdout[key] = mae
            except Exception as e:
                key = f"{cid}" if args.repeats == 1 else f"{cid}_rep{rep}"
                failures[key] = str(e)

    values = list(mae_by_holdout.values())
    if not values:
        raise RuntimeError(f"Nessun MAE raccolto. Failures: {failures}")

    result = {
        "mean_mae": mean(values),
        "mae_by_holdout": mae_by_holdout,
        "failures": failures,
        "trial_config": str(trial_cfg_path),
        "holdout_range": args.holdout_range,
        "repeats": args.repeats,
    }
    save_json(out_dir / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
