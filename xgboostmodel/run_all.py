import os
import sys
import subprocess
import time

# Configurazione manuale se config.py fallisce
try:
    from config import HOLDOUT_CID
except ImportError:
    HOLDOUT_CID = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "clients_data")
client_ps = []


def main():
    if not os.path.exists("logs"):
        os.makedirs("logs")

    print("--- AVVIO SISTEMA FEDERATED ---")

    for client_id in range(9):
        if client_id == HOLDOUT_CID:
            continue

        csv_path = os.path.join(DATA_DIR, f"group{client_id}_merged_clean.csv")
        SERVER_ADDR = "127.0.0.1:8080"
        cmd = [sys.executable, "client_app.py", str(client_id), csv_path, SERVER_ADDR]

        # >>>> QUESTA È LA FIX PER IL DUMP <<<<
        # Scriviamo stdout/stderr su file invece che nel terminale
        log_file = open(f"logs/client_{client_id}.log", "w")

        p = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )
        client_ps.append(p)
        print(f" -> Avviato Client {client_id} (Logs in logs/client_{client_id}.log)")
        time.sleep(0.5)

    try:
        print("\nIn attesa che i client finiscano...")
        for p in client_ps:
            p.wait()
    except KeyboardInterrupt:
        print("\nInterruzione! Chiudo i processi...")
        for p in client_ps:
            p.terminate()


if __name__ == "__main__":
    main()