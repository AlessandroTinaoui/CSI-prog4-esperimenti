# server_flwr.py
import flwr as fl
from config import SERVER_ADDRESS, NUM_ROUNDS
from strategy import RandomForestAggregation

def main():
    strategy = RandomForestAggregation(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=2,
        min_evaluate_clients=2,

        # Feature selection params
        missing_threshold=0.4,
        var_threshold=1e-8,
        eps_l2=0.0,
        min_clients_ratio=0.8,
    )

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
