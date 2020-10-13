import itertools
import bisect
import numpy as np
import matplotlib.pyplot as plt
from zquantum.core.bitstring_distribution import BitstringDistribution


#Global Variables
Precision = 8 
DType = np.float32 
prob_data_cutoff = 1e-8


def intToising(num, n_spins): 
    '''Converts an integer into a 1D vector of Ising variables +\-1 
    Args: 
        num (int): positive number to be converted into its corresponding Ising representation 
        n_spins (int): positive number of spins in the Ising system
    Returns: 
        1D Array of Ising varaibles +\-1 '''
    
    assert(num < 2**n_spins) 
    s = format(num, '0' + str(n_spins) + 'b' ) 
    vector = np.asarray([2.0*int(c)-1. for c in s]) 
    return vector


def isingToint(vector): 

    '''Converts a 1D vector of Ising variables +\-1 into an integer
    Args: 
        vector (1D Array): Ising variables +\-1
    Returns:
        (int): positive number representation of Ising variables'''

    vector = (np.asarray(vector) + 1.0) / 2 
    s = ''.join(str(int(e)) for e in vector) 
    num = int(s,2) 
    return num


#Generation of random h, J, and M for h and J are arrays of random coefficients sampled from a normal distribution with zero mean and 1/sqrt(n_spins) sd 
def get_random_hj(n_spins): 
    '''Generates random h, J, and M, where h and J are arrays of random coefficients sampled from a normal distribution with zero mean and 1/sqrt(n_spins) sd 
    Args: 
        n_spins (int): positive number of spins in the Ising system
    Returns:
       h (1D Array): list of random coefficients sampled from a normal distribution with zero mean and 1/sqrt(n_spins) sd
       J (Array): n_spin x n_spin array of coefficients sampled from a normal distribution with zero mean and 1/sqrt(n_spins) sd
       M (Array): n_spin x n_spin x n_spin array of coefficients sampled from a normal distribution with zero mean and 1/sqrt(n_spins) sd
       '''

    h = np.zeros((n_spins))
    J = np.zeros((n_spins,n_spins))
    M = np.zeros((n_spins,n_spins,n_spins))   
    for i in range(n_spins):
        h[i] = np.random.normal(0, 1./np.sqrt(n_spins))
        for j in range(i, n_spins):
            if i==j: continue
            J[i,j] = J[j,i] = np.random.normal(0, 1./np.sqrt(n_spins))
            for k in range(j, n_spins):
                if k==i or k==j: continue
                M[i,j,k] = M[i,k,j] = M[j,i,k] = M[j,k,i] = M[k,i,j] = M[k,j,i] = 0
    return h, J, M


def get_thermal_states_target_distribution(n_spins, beta, n_body_interaction): 
    '''Generates coherent thermal states target data
    Args: 
        n_spins (int): positive number of spins in the Ising system
        beta (int): parameter in energy expression
        n_body_interaction (int): paramter in energy expression for interacting spins 
    Returns:
       probabilities (dict): keys are binary string representations and values are corresponding probabilities (floats)
       '''
    Z = 0
    h, J, M = get_random_hj(n_spins)
    n_args = dict() 
    local_exp = np.zeros((n_spins)) 
    pair_exp = np.zeros((n_spins, n_spins), dtype = float) 
    for num in range(int(2 ** n_spins)): 
        vector = intToising(num, n_spins) 
        energy = 0 
        for i in range(n_spins): 
            energy -= vector[i] * h[i] 
            for j in range(i, n_spins): 
                if i ==j: continue 
                energy -= vector[i] * vector[j] * (J[i,j] + J[j,i]) / 2
                for k in range(j, n_spins): 
                    if k == i or k == j: continue 
                    energy -= vector[i] * vector[j] * vector[k] * (M[i,j,k]+ M[i,k,j] + M[j,i,k]+M[j,k,i]+  M[k,i,j]+ M[k,j,i])/6 
        energy -= n_body_interaction * np.prod(vector)
        energy = round(energy, Precision) 
        factor = np.exp(-energy*beta)
        Z += factor
        local_exp += factor * vector 
        pair_exp += factor * np.outer(vector, vector)
        binary = format(num, '0' + str(n_spins) + 'b' ) 
        n_args[binary] = energy 
    energies = n_args
    probabilities = {k: np.exp(-v * beta) / Z for k,v in n_args.items()} 
    local_exp = local_exp / Z
    pair_exp = pair_exp / Z
    assert(round(sum(probabilities.values()), Precision) == 1.)  
    return BitstringDistribution(probabilities)







            

                
        
        
                
        
        
            
            
        
        
