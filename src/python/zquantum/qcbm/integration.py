from cost_function import QCBMCostFunction
from ansatz import get_qcbm_ansatz, generate_random_initial_params
from zquantum.core.circuit import (
    load_circuit_template_params,
    save_circuit_template_params,
)
from zquantum.optimizers.utils import save_optimization_results
from zquantum.core.bitstring_distribution import (
    load_bitstring_distribution,
    BitstringDistribution,
)
from target import get_bars_and_stripes_target_distribution
from zquantum.core.utils import get_func_from_specs
from qeqiskit.simulator import QiskitSimulator
from zquantum.optimizers import CMAESOptimizer, ScipyOptimizer

n_qubits = 4
n_layers = 2
topology = "all"
epsilon = 1e-8
distance_measure = get_func_from_specs(
    {
        "module_name": "zquantum.core.bitstring_distribution",
        "function_name": "compute_clipped_negative_log_likelihood",
    }
)

ansatz = get_qcbm_ansatz(n_qubits, n_layers, topology)

backend = QiskitSimulator("statevector_simulator")

# optimizer = CMAESOptimizer(options={"popsize": 5, "sigma_0": 0.1, "tolx": 1e-5})
optimizer = ScipyOptimizer(method="L-BFGS-B", options={"keep_value_history": True})

import numpy as np

initial_params = generate_random_initial_params(
    n_qubits, n_layers, topology, -np.pi / 2, np.pi / 2
)
target_distribution = get_bars_and_stripes_target_distribution(2, 2)

cost_function = QCBMCostFunction(
    ansatz, backend, distance_measure, target_distribution, epsilon
)
opt_results = optimizer.minimize(cost_function, initial_params)

save_optimization_results(opt_results, "optimization-results.json")
save_circuit_template_params(opt_results.opt_params, "optimized-parameters.json")
