################################################################################
# © Copyright 2020-2021 Zapata Computing Inc.
################################################################################
import json

from zquantum.core.distribution import load_measurement_outcome_distribution
from zquantum.core.serialization import (
    load_array,
    save_array,
    save_optimization_results,
)
from zquantum.core.utils import create_object, get_func_from_specs
from zquantum.qcbm.ansatz import QCBMAnsatz
from zquantum.qcbm.cost_function import QCBMCostFunction


def optimize_variational_qcbm_circuit(
    distance_measure_specs,
    distance_measure_parameters,
    n_layers,
    n_qubits,
    n_samples,
    topology,
    backend_specs,
    optimizer_specs,
    initial_parameters,
    target_distribution,
    keep_history,
    gradient_type="finite_difference",
    gradient_kwargs=None,
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

    initial_parameters = load_array(initial_parameters)
    target_distribution = load_measurement_outcome_distribution(target_distribution)

    if isinstance(distance_measure_parameters, str):
        distance_measure_parameters = json.loads(distance_measure_parameters)

    cost_function = QCBMCostFunction(
        ansatz,
        backend,
        n_samples,
        distance_measure,
        distance_measure_parameters,
        target_distribution,
        gradient_type,
        gradient_kwargs,
    )
    opt_results = optimizer.minimize(cost_function, initial_parameters, keep_history)

    save_optimization_results(opt_results, "qcbm-optimization-results.json")
    save_array(opt_results.opt_params, "optimized-parameters.json")
