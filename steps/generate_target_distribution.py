from zquantum.qcbm.target import get_bars_and_stripes_target_distribution as _get_bars_and_stripes_target_distribution
from zquantum.core.bitstring_distribution import save_bitstring_distribution
from zquantum.core.bitstring_distribution import save_bitstring_distribution_set
from zquantum.qcbm.target_thermal_states import get_thermal_states_target_distribution as _get_thermal_states_target_distribution
from zquantum.core.bitstring_distribution import load_bitstring_distribution_set



def get_bars_and_stripes_target_distribution(nrows, ncols, fraction, method):
    distribution = _get_bars_and_stripes_target_distribution(nrows, ncols, fraction, method)

    save_bitstring_distribution(distribution, "distribution.json")

def get_thermal_states_target_distribution(n_spins, beta, n_body_interaction):
    distribution = _get_thermal_states_target_distribution(n_spins, beta, n_body_interaction)

    save_bitstring_distribution(distribution, "distribution.json")


def get_thermal_target_training_set(n_spins, beta, n_body_interaction, instances): 
    training_data_set_distributions = [ _get_thermal_states_target_distribution(n_spins, beta, n_body_interaction) for _ in range(instances) ]
    
    save_bitstring_distribution_set(training_data_set_distributions, "distribution_set.json" )




