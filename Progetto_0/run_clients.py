from __future__ import annotations

import os
import sys
import time
import signal
import argparse
import subprocess
from datetime import datetime


def project_root() -> str:
    # .../Progetto_0 -> .../ (root progetto)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, ".."))


def ensure_exists(path: str, kind: str = "path") -> None:
    if kind == "file" and not os.path.isfile(path):
        raise FileNotFoundError(f"[ERROR] File non trovato: {path}")
    if kind == "dir" and not os.path.isdir(path):
        raise FileNotFoundError(f"[ERROR] Directory non trovata: {path}")


def spawn(cmd: list[str], cwd: str, log_path: str) -> subprocess.Popen:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, "a", buffering=1)
    f.write("[CMD] " + " ".join(cmd) + "\n\n")
    return subprocess.Popen(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="8080")
    ap.add_argument("--num_clients", type=int, default=9)
    ap.add_argument("--rounds", type=int, default=6)      # 1 FS + 5 train
    ap.add_argument("--k_top", type=int, default=30)
    ap.add_argument("--max_trees", type=int, default=400)
    ap.add_argument("--start_delay", type=float, default=1.5)
    args = ap.parse_args()

    ROOT = project_root()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # scripts
    server_py = os.path.join(ROOT, "Progetto_0", "server_flwr.py")
    client_py = os.path.join(ROOT, "Progetto_0", "client_flwr.py")

    # dati client
    data_dir = os.path.join(ROOT, "dataset", "CSV_train", "CSV_train_clean")

    # log
    log_dir = os.path.join(ROOT, "logs_flower")
    os.makedirs(log_dir, exist_ok=True)

    ensure_exists(server_py, "file")
    ensure_exists(client_py, "file")
    ensure_exists(data_dir, "dir")

    server_addr = f"{args.host}:{args.port}"

    print(f"[INFO] ROOT      = {ROOT}")
    print(f"[INFO] SERVER    = {server_addr}")
    print(f"[INFO] DATA_DIR  = {data_dir}")
    print(f"[INFO] LOG_DIR   = {log_dir}")
    print("")

    # ---- start server ----
    server_log = os.path.join(log_dir, f"{stamp}_server.log")
    server_cmd = [
        sys.executable, server_py,
        "--host", args.host,
        "--port", str(args.port),
        "--num_clients", str(args.num_clients),
        "--rounds", str(args.rounds),
        "--k_top", str(args.k_top),
        "--max_trees", str(args.max_trees),
    ]
    print(f"[START] server -> {server_log}")
    p_server = spawn(server_cmd, cwd=ROOT, log_path=server_log)

    # attendo un attimo che il server si metta in ascolto
    time.sleep(args.start_delay)

    # ---- start clients ----
    client_procs: list[tuple[str, subprocess.Popen, str]] = []
    for i in range(args.num_clients):
        gid = f"group{i}"
        csv_path = os.path.join(data_dir, f"{gid}_merged_clean.csv")
        ensure_exists(csv_path, "file")

        client_log = os.path.join(log_dir, f"{stamp}_{gid}.log")
        client_cmd = [
            sys.executable, client_py,
            "--client_id", gid,
            "--csv", csv_path,
            "--server", server_addr,
        ]
        print(f"[START] {gid} -> {client_log}")
        p = spawn(client_cmd, cwd=ROOT, log_path=client_log)
        client_procs.append((gid, p, client_log))
        time.sleep(0.25)

    print("\n[INFO] Tutto avviato. Ctrl+C per terminare.\n")

    def shutdown():
        # termina prima i client, poi server
        for _, p, _ in client_procs:
            if p.poll() is None:
                p.terminate()
        time.sleep(1.0)
        for _, p, _ in client_procs:
            if p.poll() is None:
                p.kill()

        if p_server.poll() is None:
            p_server.terminate()
            time.sleep(1.0)
        if p_server.poll() is None:
            p_server.kill()

    try:
        while True:
            # se server muore -> chiudi tutto e stampa log
            if p_server.poll() is not None and p_server.returncode != 0:
                print(f"[CRASH] Server terminato (exit={p_server.returncode}). Log: {server_log}")
                shutdown()
                raise SystemExit(1)

            # se un client muore -> chiudi tutto e stampa quale
            for gid, p, logp in client_procs:
                rc = p.poll()
                if rc is not None and rc != 0:
                    print(f"[CRASH] {gid} terminato (exit={rc}). Log: {logp}")
                    shutdown()
                    raise SystemExit(1)

            # se tutti i client finiscono, aspetta server (o termina)
            if all(p.poll() is not None for _, p, _ in client_procs):
                print("[DONE] Tutti i client hanno terminato. Chiudo il server.")
                shutdown()
                break

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C ricevuto, termino tutto...")
        shutdown()
        print("[INFO] Terminato.")


if __name__ == "__main__":
    main()
