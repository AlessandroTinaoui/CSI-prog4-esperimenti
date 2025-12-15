# server/server_flwr.py
import flwr as fl
from server.config import SERVER_ADDRESS, NUM_ROUNDS
from server.strategy import RandomForestAggregation  # Importa la tua strategia personalizzata

def main():
    """Avvia il server Flower con la strategia e configurazione scelte."""
    strategy = RandomForestAggregation(
        fraction_fit=1,  # percentuale di client che partecipano all'allenamento
        fraction_evaluate=1,  # percentuale di client che partecipano alla valutazione
        min_fit_clients=2,
        min_evaluate_clients=2
    )

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
