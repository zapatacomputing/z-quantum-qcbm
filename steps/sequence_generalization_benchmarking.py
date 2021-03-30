import numpy as np
import json
from zquantum.core.utils import create_object
from zquantum.qcbm.cost_function import QCBMCostFunction
from zquantum.qcbm.ansatz import QCBMAnsatz
from zquantum.core.circuit import (
    load_circuit_template_params,
    save_circuit_template_params,
)
from zquantum.core.utils import create_object, get_func_from_specs
from zquantum.core.bitstring_distribution import load_bitstring_distribution, compute_mmd
from zquantum.qcbm.target import (
    get_bars_and_stripes_target_distribution as _get_bars_and_stripes_target_distribution,
)
from zquantum.core.serialization import save_optimization_results
from zquantum.core.bitstring_distribution import save_bitstring_distribution

nrows = 2
ncols = 2
fraction = 1.0

random_representation = _get_bars_and_stripes_target_distribution(nrows, ncols, "random", fraction)
zigzag_representation = _get_bars_and_stripes_target_distribution(nrows, ncols, "zigzag", fraction)

def test_random(test_string): 
    
    for key in random_representation.distribution_dict: 
        if key == test_string: 
            random_representation.distribution_dict[key] = 0.0
        else: 
            random_representation.distribution_dict[key] = 1 / 6
    return random_representation


def test_zigzag(test_string): 
    for key in zigzag_representation.distribution_dict: 
        if key == test_string: 
            zigzag_representation.distribution_dict[key] = 0.0
        else: 
            zigzag_representation.distribution_dict[key] = 1 / 6
    return zigzag_representation


def get_results():

    ansatz_specs = '{"module_name": "zquantum.qcbm.ansatz", "function_name": "QCBMAnsatz", "number_of_layers": 2, "number_of_qubits": 4, "topology": "all"}'
    min_value = -1.57
    max_value = 1.57
    seed = 9
    number_of_parameters = "None"
    if ansatz_specs != "None":
        ansatz_specs_dict = json.loads(ansatz_specs)
        ansatz = create_object(ansatz_specs_dict)
        number_of_params = ansatz.number_of_params
    elif number_of_parameters != "None":
        number_of_params = number_of_parameters
    if seed != "None":
        np.random.seed(seed)
    initial_parameters = np.random.uniform(min_value, max_value, number_of_params)

    n_qubits = 4
    n_layers = 2
    topology = "all"
    distance_measure_specs = '{"module_name": "zquantum.core.bitstring_distribution", "function_name": "compute_mmd"}'
    distance_measure_parameters = '{"epsilon": 1e-6}'
    backend_specs = '{"module_name": "qequlacs.simulator", "function_name": "QulacsSimulator"}'
    optimizer_specs = '{"module_name": "zquantum.optimizers.cma_es_optimizer", "function_name": "CMAESOptimizer", "options": {"popsize": 5, "sigma_0": 0.1, "tolx": 0.000001, "seed": 9}'


    distance_measure = get_func_from_specs(json.loads(distance_measure_specs))
    ansatz = QCBMAnsatz(n_layers, n_qubits, topology)
    backend = create_object(json.loads(backend_specs))
    optimizer = create_object(json.loads(optimizer_specs))

    generation_dict = {}

    for i in range(len(list(zigzag_representation.distribution_dict.keys()))): 
        cost_function = QCBMCostFunction(
            ansatz,
            backend,
            distance_measure,
            json.loads(distance_measure_parameters),
            test_zigzag(list(zigzag_representation.distribution_dict.keys())[i]),
            gradient_type = "finite_difference"
        )

        opt_results = optimizer.minimize(cost_function, initial_parameters)
        circuit = ansatz.get_executable_circuit(opt_results.opt_params)
        qcbm_data = backend.get_bitstring_distribution(circuit)
        generation_prob = qcbm_data.distribution_dict[list(zigzag_representation.distribution_dict.keys())[i]]
        generation_dict[list(zigzag_representation.distribution_dict.keys())[i]] = generation_prob 
        save_optimization_results(opt_results, "qcbm-optimization-results.json" + str(i))
        save_circuit_template_params(opt_results.opt_params, "optimized-parameters.json" + str(i))
        



