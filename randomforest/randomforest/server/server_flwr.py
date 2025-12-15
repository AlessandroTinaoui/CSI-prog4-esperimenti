# server/server_flwr.py
import flwr as fl
from config import SERVER_ADDRESS, NUM_ROUNDS
from strategy import RandomForestAggregation  # Strategia custom (con FS + RF aggregation)

def main():
    """Avvia il server Flower con la strategia e configurazione scelte."""

    strategy = RandomForestAggregation(
        # --- Federated Feature Selection (Round 1) ---
        top_k=15,  # <-- consigliato per Garmin (puoi mettere 15/20)
        save_path="selected_features.json",

        # --- Flower / Federated config ---
        fraction_fit=1.0,           # usa tutti i client disponibili ad ogni round
        fraction_evaluate=1.0,
        min_fit_clients=9,
        min_evaluate_clients=9,
        min_available_clients=9,    # <-- consigliato per evitare round con pochi client
    )

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
