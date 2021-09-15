import logging
import typing
import numpy as np
import qe.sdk.v1 as qe
from qequlacs.simulator import QulacsSimulator
from zquantum.core.bitstring_distribution import (
    BitstringDistribution,
    compute_clipped_negative_log_likelihood,
)
from zquantum.core.interfaces.optimizer import Optimizer, OptimizeResult
from zquantum.optimizers.cma_es_optimizer import CMAESOptimizer
from zquantum.qcbm.target import get_bars_and_stripes_target_distribution
from zquantum.qcbm.ansatz import QCBMAnsatz
from zquantum.qcbm.cost_function import QCBMCostFunction


@qe.step(
    resource_def=qe.ResourceDefinition(cpu="1000m", memory="2Gi", disk="10Gi"),
    custom_name="generate-init-params",
)
def generate_random_initial_parameters(
    min_value: float, max_value: float, number_of_parameters: int
) -> np.ndarray:
    return np.random.uniform(min_value, max_value, number_of_parameters)


@qe.step(
    resource_def=qe.ResourceDefinition(cpu="1000m", memory="1Gi", disk="10Gi"),
    custom_name="generate-target-distribution",
)
def generate_target_distribution(
    nrows: int, ncols: int, fraction: float = 1.0, method: str = "zigzag"
) -> BitstringDistribution:
    return get_bars_and_stripes_target_distribution(nrows, ncols, fraction, method)


@qe.step(
    resource_def=qe.ResourceDefinition(cpu="1000m", memory="2Gi", disk="10Gi"),
    custom_name="optimize-circuit",
)
def optimize_circuit(
    ansatz: QCBMAnsatz,
    backend_name: str,
    optimizer: Optimizer,
    initial_parameters: np.ndarray,
    target_distribution: BitstringDistribution,
) -> OptimizeResult:

    backend_map = {"qulacs": QulacsSimulator}
    cost_function = QCBMCostFunction(
        ansatz=ansatz,
        backend=backend_map[backend_name](),
        n_samples=None,
        distance_measure=compute_clipped_negative_log_likelihood,
        distance_measure_parameters={"epsilon": 1e-6},
        target_bitstring_distribution=target_distribution,
    )
    opt_results = optimizer.minimize(
        cost_function, initial_parameters, keep_history=True
    )
    return opt_results


@qe.workflow(
    name="qcbm-opt",
    import_defs=[
        qe.Z.Quantum.Core("dev"),
        qe.Z.Quantum.Optimizers("dev"),
        qe.QE.Qulacs("dev"),
        qe.QE.Qiskit("dev"),
        qe.GitImportDefinition.get_current_repo_and_branch(),
    ],
)
def qcbm_workflow(
    ansatz: QCBMAnsatz,
    backend_name: str,
    optimizer: Optimizer,
    seed: typing.Optional[int] = None,
) -> typing.List[qe.StepDefinition]:
    if seed is not None:
        np.random.seed(seed)
    return optimize_circuit(
        ansatz,
        backend_name,
        optimizer,
        generate_random_initial_parameters(-1.57, 1.57, int(ansatz.number_of_params)),
        generate_target_distribution(nrows=2, ncols=2, fraction=1.0, method="zigzag"),
    )


if __name__ == "__main__":
    optimizer_seed = rng_seed = 9
    wf: qe.WorkflowDefinition = qcbm_workflow(
        ansatz=QCBMAnsatz(number_of_layers=4, number_of_qubits=4, topology="all"),
        backend_name="qulacs",
        optimizer=CMAESOptimizer(
            options={
                "popsize": 5,
                "sigma_0": 0.1,
                "tolx": 0.000001,
                "seed": optimizer_seed,
            }
        ),
        seed=rng_seed,
    )

    wf.local_run(log_level=logging.INFO)
    wf.validate()
    wf.print_workflow()
    wf.submit()
