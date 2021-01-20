from zquantum.qcbm.cost_function import QCBMCostFunction
from zquantum.qcbm.ansatz import QCBMAnsatz
from zquantum.core.circuit import (
    load_circuit_template_params,
    save_circuit_template_params,
)
from zquantum.core.utils import create_object, get_func_from_specs
from zquantum.core.bitstring_distribution import load_bitstring_distribution_set
import json


def cost_function_set(
    distance_measure_specs,
    distance_measure_parameters,
    n_layers,
    n_qubits,
    topology,
    backend_specs,
    optimizer_specs,
    initial_parameters,
    target_distribution_list,
):

    distance_measure = get_func_from_specs(json.loads(distance_measure_specs))
    ansatz = QCBMAnsatz(n_layers, n_qubits, topology)
    backend = create_object(json.loads(backend_specs))
    optimizer = create_object(json.loads(optimizer_specs))
    initial_parameters = load_circuit_template_params(initial_parameters)
    target_distribution_list = load_bitstring_distribution_set(target_distribution_list)
    
    
    training_data_set_cost_functions = [
        QCBMCostFunction(
            ansatz,
            backend,
            distance_measure,
            json.loads(distance_measure_parameters),
            target_distribution_list,
        ) for target_bitstring_distribution in target_distribution_list
]
    #Need to create function save_cost_function_set to save cost functions to a json file 
    #Also am going to need a function that loads them 
    #Same for Generic Problem 
    save_cost_function_set(training_data_set_cost_functions, "cost_function_set.json" )





