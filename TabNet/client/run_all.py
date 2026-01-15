from __future__ import annotations

import os
import sys
import subprocess
import time
from pathlib import Path

from TabNet.server.config import HOLDOUT_CID
from dataset.dataset_cfg import get_train_path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR/ ".."/".." / Path(get_train_path())).resolve()


def main():
    logs_dir = BASE_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    print(BASE_DIR)
    print(DATA_DIR)
    print("--- AVVIO SISTEMA FEDERATED (TabNet) ---")
    print(f"DATA_DIR: {DATA_DIR}")

    client_ps = []

    for client_id in range(9):
        if 0 <= HOLDOUT_CID <= 8 and client_id == HOLDOUT_CID:
            continue

        csv_path = DATA_DIR / f"group{client_id}_merged_clean.csv"
        if not csv_path.exists():
            print(f"File mancante per client {client_id}: {csv_path}")
            continue

        cmd = [sys.executable, str(BASE_DIR / "client_app.py"), str(client_id), str(csv_path)]
        log_file = open(logs_dir / f"client_{client_id}.log", "w", encoding="utf-8")

        p = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        client_ps.append((p, log_file))
        print(f" -> Avviato Client {client_id} (log: logs/client_{client_id}.log)")
        time.sleep(0.4)

    try:
        print("\nIn attesa che i client finiscano...")
        for p, lf in client_ps:
            p.wait()
            lf.close()
    except KeyboardInterrupt:
        print("\nInterruzione! Chiudo i processi...")
        for p, lf in client_ps:
            p.terminate()
            lf.close()


if __name__ == "__main__":
    main()
