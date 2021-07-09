import unittest
import numpy as np
from zquantum.core.utils import dec2bin, convert_tuples_to_bitstrings
from zquantum.core.bitstring_distribution import BitstringDistribution
from zquantum.core.bitstring_distribution.distance_measures.mmd import compute_mmd
from .target_thermal_states import (
    convert_integer_to_ising_bitstring,
    convert_ising_bitstring_to_integer,
    get_random_hamiltonian_parameters,
    get_thermal_target_distribution_dict,
    get_target_bitstring_distribution,
    cumulate,
    sample,
    get_thermal_sampled_distribution,
    get_sampled_bitstring_distribution,
)


SEED = 14943


class TestThermalTarget(unittest.TestCase):
    def test_convert_integer_to_ising_bitstring(self):
        # Given
        number = 4
        length = 5
        expected_ising_bitstring = [-1, -1, 1, -1, -1]

        # When
        ising_bitstring = convert_integer_to_ising_bitstring(number, length)

        # Then
        self.assertListEqual(ising_bitstring, expected_ising_bitstring)

        # When 
        ising_bitstring_zero = convert_integer_to_ising_bitstring(0, length)

        # Then
        self.assertListEqual(ising_bitstring_zero, [0,0,0,0,0])


    def test_convert_ising_bitstring_to_integer(self):
        # Given
        ising_bitstring = [-1, -1, 1, -1, -1]
        expected_number = 4
        zero_bitstring = [0, 0, 0, 0, 0]

        # When
        number = convert_ising_bitstring_to_integer(ising_bitstring)

        # Then
        self.assertEqual(number, expected_number)

        # When
        number = convert_ising_bitstring_to_integer(zero_bitstring)

        # Then
        self.assertEqual(number, 0)



    def test_ising2int2ising(self):
        # Given
        expected_number = 14
        length = 5
        zero_number = 0 
        zero_length = 0
        # When
        number = convert_ising_bitstring_to_integer(convert_integer_to_ising_bitstring(expected_number, length))

        # Then
        self.assertEqual(number, expected_number)

        # When
        number = convert_ising_bitstring_to_integer(convert_integer_to_ising_bitstring(zero_number, length))

        # Then
        self.assertEqual(number, zero_number)

        # When
        number = convert_ising_bitstring_to_integer(convert_integer_to_ising_bitstring(zero_number, zero_length))

        # Then
        self.assertEqual(number, zero_number)



    def test_get_random_hamiltonian_parameters(self):
        # Given
        np.random.seed(seed=SEED)
        n_spins = 15

        # When
        hamiltonian_parameters = get_random_hamiltonian_parameters(n_spins)

        # Then
        self.assertEqual(
            len(hamiltonian_parameters[0]),
            n_spins,
        )
        self.assertEqual(
            hamiltonian_parameters[1].shape,
            (n_spins, n_spins),
        )

    def test_get_thermal_target_distribution_dict(self):
        # Given
        n_spins = 5 
        temperature = 1.0
        expected_bitstrings = convert_tuples_to_bitstrings(
            [dec2bin(number, n_spins) for number in range(int(2 ** n_spins))]
        )
        expected_keys = expected_bitstrings[len(expected_bitstrings) :: -1]
        np.random.seed(SEED)
        external_fields = np.random.rand(n_spins)
        two_body_couplings = np.random.rand(n_spins, n_spins)
        hamiltonian_parameters = [external_fields, two_body_couplings]

        # When
        target_distribution = get_thermal_target_distribution_dict(
            n_spins, temperature, hamiltonian_parameters
        )

        # Then
        self.assertListEqual(
            sorted(list(target_distribution.keys())), sorted(expected_keys)
        )

        self.assertAlmostEqual(
            sum(list(target_distribution.values())),
            1.0,
        )

    def test_get_target_bitstring_distribution(self):
        # Given
        np.random.seed(SEED)
        n_spins = 6
        temperature = 1.0
        external_fields = np.random.rand(n_spins)
        two_body_couplings = np.random.rand(n_spins, n_spins)
        hamiltonian_parameters = [external_fields, two_body_couplings]

        target_distribution = get_thermal_target_distribution_dict(
            n_spins, temperature, hamiltonian_parameters
        )

        # When
        np.random.seed(SEED)
        target_bitstring_distribution = get_target_bitstring_distribution(
            n_spins, temperature, hamiltonian_parameters
        )

        # Then
        self.assertTrue(
            isinstance(target_bitstring_distribution, BitstringDistribution)
        )
        print()
        self.assertDictEqual(
            target_bitstring_distribution.distribution_dict, target_distribution
        )

    def test_cumulate(self):
        # Given
        n_spins = 3
        temperature = 2.0
        np.random.seed(SEED)
        external_fields = np.random.rand(n_spins)
        two_body_couplings = np.random.rand(n_spins, n_spins)
        hamiltonian_parameters = [external_fields, two_body_couplings]

        # When
        cumulative_probabilities = cumulate(
            n_spins, temperature, hamiltonian_parameters
        )

        # Then
        self.assertAlmostEqual(cumulative_probabilities[-1], 1.0)

    def test_sample(self):
        # Given
        n_samples = 1000
        n_spins = 2
        temperature = 0.7
        np.random.seed(SEED)
        external_fields = np.random.rand(n_spins)
        two_body_couplings = np.random.rand(n_spins, n_spins)
        hamiltonian_parameters = [external_fields, two_body_couplings]

        # When
        samples = sample(n_samples, n_spins, temperature, hamiltonian_parameters)

        # Then
        self.assertEqual(samples.shape, (n_samples, n_spins))

    def test_thermal_sampled_distribution(self):
        # Given
        n_samples = 5000
        n_spins = 2
        temperature = 0.85
        np.random.seed(SEED)
        external_fields = np.random.rand(n_spins)
        two_body_couplings = np.random.rand(n_spins, n_spins)
        hamiltonian_parameters = [external_fields, two_body_couplings]
        expected_bitstrings = convert_tuples_to_bitstrings(
            [dec2bin(number, n_spins) for number in range(int(2 ** n_spins))]
        )
        expected_keys = expected_bitstrings[len(expected_bitstrings) :: -1]

        # When
        sample_distribution = get_thermal_sampled_distribution(
            n_samples, n_spins, temperature, hamiltonian_parameters
        )[0]

        # Then
        self.assertListEqual(
            sorted(list(sample_distribution.keys())), sorted(expected_keys)
        )

        self.assertAlmostEqual(sum(list(sample_distribution.values())), 1)

    def test_get_sampled_bitstring_distribution(self):
        # Given
        np.random.seed(SEED)
        n_samples = 1000
        n_spins = 8
        temperature = 1.2
        external_fields = np.random.rand(n_spins)
        two_body_couplings = np.random.rand(n_spins, n_spins)
        hamiltonian_parameters = [external_fields, two_body_couplings]
        np.random.seed(SEED)
        target_distribution = get_thermal_sampled_distribution(
            n_samples, n_spins, temperature, hamiltonian_parameters
        )[0]

        # When
        np.random.seed(SEED)
        target_bitstring_distribution = get_sampled_bitstring_distribution(
            n_samples, n_spins, temperature, hamiltonian_parameters
        )[0]

        # Then
        self.assertTrue(
            isinstance(target_bitstring_distribution, BitstringDistribution)
        )
        print()
        self.assertDictEqual(
            target_bitstring_distribution.distribution_dict, target_distribution
        )

    def test_samples_from_distribution(self):
        # Given
        n_samples = 10000
        n_spins = 4
        temperature = 1.0
        distance_measure = {"epsilon": 1e-6}
        np.random.seed(SEED)
        external_fields = np.random.rand(n_spins)
        two_body_couplings = np.random.rand(n_spins, n_spins)
        hamiltonian_parameters = [external_fields, two_body_couplings]
        np.random.seed(SEED)
        actual = get_target_bitstring_distribution(
            n_spins, temperature, hamiltonian_parameters
        )
        np.random.seed(SEED)
        model = get_sampled_bitstring_distribution(
            n_samples, n_spins, temperature, hamiltonian_parameters
        )[0]

        # When
        mmd = compute_mmd(actual, model, distance_measure)

        # Then
        self.assertLess(mmd, 1e-4)


if __name__ == "__main__":
    unittest.main()
