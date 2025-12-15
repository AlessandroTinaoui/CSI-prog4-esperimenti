import pickle
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import Parameters, FitRes, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import numpy as np

class RandomForestAggregation(FedAvg):
    """Strategia che aggrega i modelli Random Forest combinando gli alberi."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggrega i modelli Random Forest combinando gli alberi."""
        
        if not results:
            return None, {}

        all_models = []
        total_examples = 0
        for client_proxy, fit_res in results:
            num_examples = fit_res.num_examples
            total_examples += num_examples
            
            # Deserializza il modello
            params = parameters_to_ndarrays(fit_res.parameters)
            
            if len(params) > 0 and params[0].size > 0:
                try:
                    # Deserializza e aggiungi il modello alla lista
                    model_bytes = params[0].tobytes()
                    model = pickle.loads(model_bytes)
                    all_models.append((model, num_examples))
                except Exception as e:
                    continue
        
        # Combinazione degli alberi dei modelli
        combined_model = all_models[0][0]
        all_estimators = []
        for model, _ in all_models:
            all_estimators.extend(model.estimators_)
        
        combined_model.estimators_ = all_estimators
        combined_model.n_estimators = len(all_estimators)
        
        # Serializza il modello aggregato
        model_bytes = pickle.dumps(combined_model)
        model_array = np.frombuffer(model_bytes, dtype=np.uint8)
        parameters = ndarrays_to_parameters([model_array])
        
        # Aggrega le metriche di allenamento
        metrics_aggregated = {}
        if results:
            total_acc = sum([fit_res.metrics.get("train_accuracy", 0) * fit_res.num_examples 
                           for _, fit_res in results])
            avg_accuracy = total_acc / total_examples if total_examples > 0 else 0
            
            metrics_aggregated = {
                "train_accuracy": avg_accuracy,
                "total_trees": combined_model.n_estimators,
            }
        
        return parameters, metrics_aggregated
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Dict[str, Scalar]:
        """Aggrega i risultati della valutazione."""
        
        if not results:
            return {}

        total_examples = 0
        total_eval_accuracy = 0
        
        # Sum all evaluation accuracies from the clients
        for client_proxy, fit_res in results:
            num_examples = fit_res.num_examples
            total_examples += num_examples
            print(f"Client {client_proxy.cid} evaluated on {num_examples} examples.")
            # Aggregating evaluation accuracy (eval_accuracy)
            eval_accuracy = fit_res.metrics.get("eval_accuracy", 0)
            total_eval_accuracy += eval_accuracy * num_examples  # Weighted average based on the number of examples
        
        # Calculate average evaluation accuracy
        avg_eval_accuracy = total_eval_accuracy / total_examples if total_examples > 0 else 0
        print(f"Total evaluation examples: {total_examples}")
        # Return the aggregated evaluation metrics
        metrics_aggregated = {
            "eval_accuracy": avg_eval_accuracy
        }
        
        print(f"Aggregated evaluation accuracy: {avg_eval_accuracy:.4f}")
        
        return 0, metrics_aggregated
