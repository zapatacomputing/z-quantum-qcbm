import itertools
import bisect
import numpy as np
import matplotlib.pyplot as plt
from zquantum.core.bitstring_distribution import BitstringDistribution

#Global Variables
Precision = 8 
DType = np.float32 
prob_data_cutoff = 1e-8
CLIP = 1e-6
distance_measure_params = {"epsilon": 1e-6}


def intToising(num, n_spins): 
    '''Converts an integer into a 1D vector of Ising variables +-1 
    Args: 
        num (int): positive number to be converted into its corresponding Ising representation 
        n_spins (int): positive number of spins in the Ising system
    Returns: 
        1D Array of Ising varaibles +-1 '''
    
    assert(num < 2**n_spins) 
    s = format(num, '0' + str(n_spins) + 'b' ) 
    vector = np.asarray([2.0*int(c)-1. for c in s]) 
    return vector


def isingToint(vector): 

    '''Converts a 1D vector of Ising variables +-1 into an integer
    Args: 
        vector (1D Array): Ising variables +\-1
    Returns:
        (int): positive number representation of Ising variables'''

    vector = (np.asarray(vector) + 1.0) / 2 
    s = ''.join(str(int(e)) for e in vector) 
    num = int(s,2) 
    return num



def get_random_hj(n_spins, seed_dist ): 
    '''Generates random h, J, and where h and J are arrays of random coefficients sampled from a normal distribution with zero mean and sqrt(n_spins) sd 
    Args: 
        n_spins (int): positive number of spins in the Ising system
        seed_dist (int): seed integer in order to keep track of the created random target distributions 
    Returns:
       h (1D Array): list of random coefficients sampled from a normal distribution with zero mean and sqrt(n_spins) sd
       J (Array): n_spin x n_spin array of coefficients sampled from a normal distribution with zero mean and sqrt(n_spins) sd
       '''
    np.random.seed(seed_dist)
    h = np.zeros((n_spins))
    J = np.zeros((n_spins,n_spins))
    for i in range(n_spins):
        h[i] = np.random.normal(0, np.sqrt(n_spins))
        for j in range(i, n_spins):
            if i==j: continue
            J[i,j] = J[j,i] = np.random.normal(0, np.sqrt(n_spins))
    return h, J


def thermal_target_distribution(n_spins, beta, seed_dist): 
    '''Generates thermal states target distribution 
    Args: 
        n_spins (int): positive number of spins in the Ising system
        beta (int): parameter in energy expression for (1 / T) for the boltzman distribution 
        seed_dist (int): seed integer in order to keep track of the created random target distributions 
    Returns:
       probabilities (dict): keys are binary string representations and values are corresponding probabilities (floats)
       '''
    Z = 0
    h, J = get_random_hj(n_spins, seed_dist)
    print("h = " + str(h))
    print("J = " + str(J))
    n_args = dict() 
    for num in range(int(2 ** n_spins)): 
        vector = intToising(num, n_spins) 
        energy = 0 
        for i in range(n_spins): 
            energy -= vector[i] * h[i] 
            for j in range(n_spins): 
                if j == i+1: 
                    energy -= vector[i] * vector[j] * J[i,j]
                else: continue
        energy = round(energy, Precision) 
        factor = np.exp(energy*beta)
        Z += factor
        print("j = " + str(Z))
        binary = format(num, '0' + str(n_spins) + 'b' )
        reverse = binary[len(binary)::-1]
        n_args[reverse] = energy 
    probabilities = {k: np.exp(v * beta) / Z for k,v in n_args.items()} 
    assert(round(sum(probabilities.values()), Precision) == 1.)  
    print(probabilities)
    return probabilities


def get_target_Bitstring_Dist(n_spins, beta, seed_dist): 
    '''Generates thermal states target data in BitStringDistribution object 

        n_spins (int): positive number of spins in the Ising system
        beta (int): parameter in energy expression for (1 / kT) for the boltzman distribution 
        seed_dist (int): seed integer in order to keep track of the created random target distributions 
    Returns:
       probabilities (BitstringDistribution): keys are binary string representations and values are corresponding probabilities (floats)
       '''
    return BitstringDistribution(thermal_target_distribution(n_spins, beta, seed_dist))



def cumulate(n_spins, beta, seed_dist):
    '''Generates array of cumulative probabilities from target_distribution
    Args: 
        n_spins (int): positive number of spins in the Ising system
        beta (int): parameter in energy expression for (1 / kT) for the boltzman distribution 
        seed_dist (int): seed integer in order to keep track of the created random target distributions  
    Returns:
       cumulative (1D array): elements are cumulative probabilities from the target distribution (floats)
       '''
    probabilities = thermal_target_distribution(n_spins, beta, seed_dist)
    assert(probabilities)
    cumulative = []
    tot = 0.
    for i in range(len(probabilities.keys())):
        tot += list(probabilities.values())[i]
        cumulative.append(tot)
    return cumulative 


#sample from the inverse cumulative partition function
def sample(n_samples, n_spins, beta, seed_dist, seed_sample):
    '''Generates samples from the original target distribution in the form of a list of ising vectors
    Args: 
        n_samples (int): the number of samples from the original distribution
        n_spins (int): positive number of spins in the Ising system
        beta (int): parameter in energy expression for (1 / kT) for the boltzman distribution 
        seed_dist (int): seed integer in order to keep track of the created random target distributions 
        seed_sample (int): seed intin order to keep track of the random distribution samples 
    Returns:
       samples (2D asarray): array of sample n_spin ising vectors 
       '''
    cumulative = cumulate(n_spins, beta, seed_dist)
    samples = np.zeros((n_samples,n_spins))
    y = np.zeros(n_samples)
    np.random.seed(seed_sample)
    for i in range(n_samples): 
        y[i] = np.random.uniform(low=0,high=1) 
    for i in range(len(y)): 
        idx = bisect.bisect_right(cumulative, y[i])
        vector = intToising(idx, n_spins)
        samples[i,:] = vector
    return np.asarray(samples)

def thermal_sample_distribution(n_samples, n_spins, beta, seed_dist, seed_sample): 
    '''Generates thermal states sample distribution
    Args: 
        n_samples (int): the number of samples from the original distribution
        n_spins (int): positive number of spins in the Ising system
        beta (int): parameter in energy expression for (1 / kT) for the boltzman distribution 
        seed_dist (int): seed integer in order to keep track of the created random target distributions 
        seed_sample (int): seed intin order to keep track of the random distribution samples 
    Returns:
       historgram_samples (dict): keys are binary string representations and values are corresponding probabilities (floats)
       '''
    n_args = dict()
    samples = sample(n_samples, n_spins, beta, seed_dist, seed_sample)
    histogram_samples = [0 for _ in range(2**n_spins)]
    for s in samples: 
        idx = isingToint(s)
        histogram_samples[idx] += 1 / float(n_samples) 
    for num in range(int(2 ** n_spins)): 
        binary = format(num, '0' + str(n_spins) + 'b' )
        reverse = binary[len(binary)::-1]
        n_args[reverse] = histogram_samples[num]
    assert(round(sum(n_args.values()), Precision) )
    return n_args 

def get_sample_Bitstring_Dist(n_samples, n_spins, beta, seed_dist, seed_sample): 
    '''Generates thermal states sample distribution as BitstringDistribution object
    Args: 
        n_samples (int): the number of samples from the original distribution
        n_spins (int): positive number of spins in the Ising system
        beta (int): parameter in energy expression for (1 / kT) for the boltzman distribution 
        seed_dist (int): seed integer in order to keep track of the created random target distributions 
        seed_sample (int): seed intin order to keep track of the random distribution samples 
    Returns:
       historgram_samples (BitstringDistribution): keys are binary string representations and values are corresponding probabilities (floats)
       '''
    probabilities = thermal_sample_distribution(n_samples, n_spins, beta, seed_dist, seed_sample)
    return BitstringDistribution(probabilities) 

            
            
        