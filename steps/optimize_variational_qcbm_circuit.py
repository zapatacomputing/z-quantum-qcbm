from zquantum.qcbm.cost_function import QCBMCostFunction
from zquantum.qcbm.ansatz import QCBMAnsatz
from zquantum.core.circuit import (
    load_circuit_template_params,
    save_circuit_template_params,
)
from zquantum.core.utils import create_object, get_func_from_specs
from zquantum.core.serialization import save_optimization_results
from zquantum.core.bitstring_distribution import load_bitstring_distribution
import json


def optimize_variational_qcbm_circuit(
    distance_measure_specs,
    distance_measure_parameters,
    n_layers,
    n_qubits,
    topology,
    backend_specs,
    optimizer_specs,
    initial_parameters,
    target_distribution,
):

    if isinstance(distance_measure_specs, str):
        distance_measure_specs = json.loads(distance_measure_specs)
    distance_measure = get_func_from_specs(distance_measure_specs)

    ansatz = QCBMAnsatz(n_layers, n_qubits, topology)

    if isinstance(backend_specs, str):
        backend_specs = json.loads(backend_specs)
    backend = create_object(backend_specs)

    if isinstance(optimizer_specs, str):
        optimizer_specs = json.loads(optimizer_specs)
    optimizer = create_object(optimizer_specs)

    initial_parameters = load_circuit_template_params(initial_parameters)
    target_distribution = load_bitstring_distribution(target_distribution)

    if isinstance(distance_measure_parameters, str):
        distance_measure_parameters = json.loads(distance_measure_parameters)

    cost_function = QCBMCostFunction(
        ansatz,
        backend,
        distance_measure,
        distance_measure_parameters,
        target_distribution,
    )
    opt_results = optimizer.minimize(cost_function, initial_parameters)
    save_optimization_results(opt_results, "qcbm-optimization-results.json")
    save_circuit_template_params(opt_results.opt_params, "optimized-parameters.json")
