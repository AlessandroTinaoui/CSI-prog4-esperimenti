import subprocess
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

PY = sys.executable  # python del venv

# Avvia 9 client (0..8)
clients = []
for cid in range(9):
    p = subprocess.Popen([PY, "client_app.py", str(cid)])
    clients.append(p)
    print(f"Started client {cid} (pid={p.pid})")

# Aspetta che tutti i client finiscano prima di terminare
for p in clients:
    p.wait()
