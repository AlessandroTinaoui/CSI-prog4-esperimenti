import subprocess
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

# Avvia i 4 client
clients = [
    subprocess.Popen(["python", "client_app.py", "0"]),
    subprocess.Popen(["python", "client_app.py", "1"]),
    subprocess.Popen(["python", "client_app.py", "2"]),
    subprocess.Popen(["python", "client_app.py", "3"])
]

# Aspetta che tutti i client finiscano prima di terminare
for client in clients:
    client.wait()

#server.wait()  # Aspetta che il server finisca
