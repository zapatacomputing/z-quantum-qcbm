import os
import pytest
import numpy as np
from generate_target_distribution import (
    get_thermal_states_target_distribution,
    get_thermal_target_training_set,
)
from zquantum.core.bitstring_distribution import load_bitstring_distribution_set


@pytest.mark.integration_test
def test_get_thermal_states_target_distribution_works():
    # Setup
    distribution_filename = "distribution.json"
    if os.path.isfile(distribution_filename):
        os.remove(distribution_filename)
    # Given
    n_spins = 4
    temperature = 1.0
    external_fields = np.ones(n_spins)
    two_body_couplings = np.diag(np.ones(n_spins))
    hamiltonian_parameters = [external_fields, two_body_couplings]

    # When
    get_thermal_states_target_distribution(n_spins, temperature, hamiltonian_parameters)

    # Then
    assert os.path.isfile(distribution_filename)
    os.remove(distribution_filename)


@pytest.mark.integratio_test
def test_get_thermal_target_training_set_works():
    # Setup
    distribution_set_filename = "distribution_set.json"
    if os.path.isfile(distribution_set_filename):
        os.remove(distribution_set_filename)

    # Given
    n_spins = 4
    temperature = 1.0
    external_fields = np.ones(n_spins)
    two_body_couplings = np.diag(np.ones(n_spins))
    hamiltonian_parameters = [external_fields, two_body_couplings]
    number_of_instances = 50
    seed = 21673

    # When
    get_thermal_target_training_set(
        n_spins, temperature, hamiltonian_parameters, number_of_instances, seed
    )

    # Then
    assert os.path.isfile(distribution_set_filename)
    loaded_distribution_set = load_bitstring_distribution_set(distribution_set_filename)
    assert len(loaded_distribution_set) == number_of_instances
    os.remove(distribution_set_filename)
