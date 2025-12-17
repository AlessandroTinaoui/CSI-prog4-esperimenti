import os
import sys
import subprocess
from server.config import HOLDOUT_CID

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "clients_data")
client_ps = []

for client_id in range(9):
    if client_id == HOLDOUT_CID:
        print(f"Skipping client {client_id} (HOLDOUT)")
        continue

    csv_path = os.path.join(DATA_DIR, f"group{client_id}_merged_clean.csv")
    csv_path = os.path.abspath(csv_path)

    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "client_app.py"),
        str(client_id),
        csv_path,  # <-- secondo argomento posizionale
    ]
    p = subprocess.Popen(cmd, cwd=BASE_DIR)

    p = subprocess.Popen(
        cmd,
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    client_ps.append(p)

for p in client_ps:
    p.wait()

    out, err = p.communicate()
    print(out)
    print(err)

    print(f"Started client {client_id} (pid={p.pid})")
