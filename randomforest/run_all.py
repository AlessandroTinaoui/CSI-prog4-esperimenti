import subprocess
import warnings
import sys
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

PY = sys.executable
# Avvia i 4 client
clients = [
    subprocess.Popen([PY, "client_app.py", "0"]),
    subprocess.Popen([PY, "client_app.py", "1"]),
    subprocess.Popen([PY, "client_app.py", "2"]),
    subprocess.Popen([PY, "client_app.py", "3"]),
    subprocess.Popen([PY, "client_app.py", "4"]),
    subprocess.Popen([PY, "client_app.py", "5"]),
    subprocess.Popen([PY, "client_app.py", "6"]),
    subprocess.Popen([PY, "client_app.py", "7"]),
    subprocess.Popen([PY, "client_app.py", "8"]),
]

# Aspetta che tutti i client finiscano prima di terminare
for client in clients:
    client.wait()

#server.wait()  # Aspetta che il server finisca
