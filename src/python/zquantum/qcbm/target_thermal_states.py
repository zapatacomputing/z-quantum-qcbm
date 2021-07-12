import bisect
import typing
import numpy as np
from zquantum.core.bitstring_distribution import BitstringDistribution
from zquantum.core.utils import dec2bin, bin2dec, convert_tuples_to_bitstrings


def convert_integer_to_ising_bitstring(number: int, length: int) -> typing.List[int]:
    """Converts an integer into a +/-1s bitstring (also called Ising bitstring).
    Args:
        number: positive number to be converted into its corresponding Ising bitstring representation
        length: length of the Ising bitstring (i.e. positive number of spins in the Ising system)
    Returns:
        Ising bitstring representation (1D array of +/-1)."""
    
    if length < 0: 
        raise ValueError('Length cannot be negative.')
    binary_bitstring = dec2bin(number, length)
    ising_bitstring = [bit * 2 - 1 for bit in binary_bitstring]
    return ising_bitstring


def convert_ising_bitstring_to_integer(ising_bitstring: typing.List[int]) -> int:
    """Converts a +/-1s bitstring (also called Ising bitstring) into an integer.
    Args:
        ising_bitstring: 1D array of +/-1.
    Returns:
        Integer number representation of the Ising bitstring."""

    binary_bitstring = [int((bit + 1) / 2) for bit in ising_bitstring]
    number = bin2dec(binary_bitstring)
    return number


def __get_random_hamiltonian_parameters(n_spins: int) -> typing.Tuple:
    """Generates random h, J, and where h and J are arrays of random coefficients sampled from a normal distribution with zero mean and sqrt(n_spins) sd.
    For reproducibilty, fix random generator seed in the higher level from which this function is called.
    Args:
        n_spins (int): positive number of spins in the Ising system.
    Returns:
       external_fields (array): n_spin coefficients sampled from a normal distribution with zero mean and sqrt(n_spins) sd
       two_body_couplings (array): n_spin x n_spin array of coefficients sampled from a normal distribution with zero mean and sqrt(n_spins) sd
    """
    external_fields = np.zeros((n_spins))
    two_body_couplings = np.zeros((n_spins, n_spins))
    for i in range(n_spins):
        external_fields[i] = np.random.normal(0, np.sqrt(n_spins))
        for j in range(i, n_spins):
            if i == j:
                continue
            two_body_couplings[i, j] = two_body_couplings[j, i] = np.random.normal(
                0, np.sqrt(n_spins)
            )
    return external_fields, two_body_couplings


def get_thermal_target_distribution_dict(
    n_spins: int,
    temperature: float,
    hamiltonian_parameters: typing.List[np.array],  
) -> typing.Dict:
    """Generates thermal states target distribution, saved in a dict where keys are bitstrings and
    values are corresponding probabilities according to the Boltzmann Distribution formula.

    Args:
        n_spins: positive number of spins in the Ising system
        temperature: temperature factor in the boltzman distribution
        hamiltonian_parameters: values of hamiltonian parameters, namely external fields and two body couplings.

    Returns:
       Thermal target distribution.
       Number of positive spins in the spin state. 
    """
    partition_function = 0
    external_fields, two_body_couplings = hamiltonian_parameters
    beta = 1.0 / temperature
    distribution = {}

    for spin in range(int(2 ** n_spins)):
        ising_bitstring = convert_integer_to_ising_bitstring(spin, n_spins)
        energy = 0
        for i in range(n_spins):
            energy -= ising_bitstring[i] * external_fields[i]
            energy -= ising_bitstring[i] * ising_bitstring[i+1] * two_body_couplings[i, i+1]
        boltzmann_factor = np.exp(energy * beta)
        partition_function += boltzmann_factor

        binary_bitstring = convert_tuples_to_bitstrings([dec2bin(spin, n_spins)])[-1]
        distribution[binary_bitstring] = boltzmann_factor

    normalized_distribution = {
        key: value / partition_function for key, value in distribution.items()
    }

    return BitstringDistribution(normalized_distribution) 

def __get_cumulative_probability_distribution(
    n_spins: int,
    temperature: float,
    hamiltonian_parameters: typing.List[np.array], 
) -> typing.List[float]:
    """Generates array of cumulative probabilities from a target_distribution.
    Args:
        n_spins: positive number of spins in the Ising system
        temperature: temperature factor in the boltzman distribution

    Returns:
       List of cumulative probabilities from the target distribution.
    """
    probabilities = get_thermal_target_distribution_dict(
        n_spins, temperature, hamiltonian_parameters
    )
    cumulative = []
    tot = 0.0
    for probability in probabilities.values():
        tot += probability
        cumulative.append(tot)
    return cumulative


def __get_samples_from_distribution(
    n_samples: int,
    n_spins: int,
    temperature: float,
    hamiltonian_parameters: typing.List[np.array], 
) -> np.array:
    """Generates samples from the original target distribution in the form of a list of ising vectors
    Args:
        n_samples: the number of samples from the original distribution
        n_spins: positive number of spins in the Ising system
        temperature: temperature factor in the boltzman distribution
    Returns:
       samples: array of sample n_spin ising vectors
    """
    cumulative = __get_cumulative_probability_distribution(n_spins, temperature, hamiltonian_parameters)
    samples = np.zeros((n_samples, n_spins))
    y = np.zeros(n_samples)
    for i in range(n_samples):
        y[i] = np.random.uniform(low=0, high=1)
    for i in range(len(y)):
        idx = bisect.bisect_right(cumulative, y[i])
        ising_bitstring = convert_integer_to_ising_bitstring(idx, n_spins)
        samples[i, :] = ising_bitstring
    return samples



def get_thermal_sampled_distribution(
    n_samples: int,
    n_spins: int,
    temperature: float,
    hamiltonian_parameters: typing.List[np.array],  
) -> typing.Dict:
    """Generates thermal states sample distribution
    Args:
        n_samples: the number of samples from the original distribution
        n_spins: positive number of spins in the Ising system
        temperature: temperature factor in the boltzman distribution
    Returns:
       histogram_samples: keys are binary string representations and values are corresponding probabilities.
    """
    sample_distribution_dict = {}
    samples = __get_samples_from_distribution(n_samples, n_spins, temperature, hamiltonian_parameters)
    histogram_samples = np.zeros(2 ** n_spins)
    pos_spins_list = []
    for s in samples:
        idx = convert_ising_bitstring_to_integer(s)
        histogram_samples[idx] += 1.0 / n_samples
        pos_spins = 0 
        for elem in s: 
            if elem == 1.0: 
                pos_spins += 1 
        pos_spins_list.insert(idx,pos_spins)

    for spin in range(int(2 ** n_spins)):
        binary_bitstring = convert_tuples_to_bitstrings([dec2bin(spin, n_spins)])[-1]
        sample_distribution_dict[binary_bitstring] = histogram_samples[spin]

    return BitstringDistribution(sample_distribution_dict), pos_spins_list





