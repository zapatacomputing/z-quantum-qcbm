import numpy as np
from zquantum.core.bitstring_distribution import (
    save_bitstring_distribution,
    save_bitstring_distribution_set,
)
from zquantum.qcbm.target import (
    get_bars_and_stripes_target_distribution as _get_bars_and_stripes_target_distribution,
)
from zquantum.qcbm.target_thermal_states import (
    get_target_bitstring_distribution as _get_thermal_states_target_distribution,
)


def get_bars_and_stripes_target_distribution(nrows, ncols, fraction, method):
    distribution = _get_bars_and_stripes_target_distribution(
        nrows, ncols, fraction, method
    )

    save_bitstring_distribution(distribution, "distribution.json")


def get_thermal_states_target_distribution(
    n_spins, temperature, hamiltonian_parameters
):
    distribution = _get_thermal_states_target_distribution(
        n_spins, temperature, hamiltonian_parameters
    )
    save_bitstring_distribution(distribution, "distribution.json")


def get_thermal_target_training_set(
    n_spins, temperature, hamiltonian_parameters, number_of_instances, seed
):
    np.random.seed(seed=seed)
    training_data_set_distributions = [
        _get_thermal_states_target_distribution(
            n_spins, temperature, hamiltonian_parameters
        )
        for _ in range(number_of_instances)
    ]

    save_bitstring_distribution_set(
        training_data_set_distributions, "distribution_set.json"
    )
