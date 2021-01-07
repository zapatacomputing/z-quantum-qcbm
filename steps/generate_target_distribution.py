from zquantum.qcbm.target import (
    get_bars_and_stripes_target_distribution as _get_bars_and_stripes_target_distribution,
)
from zquantum.core.bitstring_distribution import save_bitstring_distribution
from zquantum.qcbm.target_thermal_states import (
    get_thermal_states_target_distribution as _get_thermal_states_target_distribution,
)


def get_bars_and_stripes_target_distribution(nrows, ncols, fraction, method):
    distribution = _get_bars_and_stripes_target_distribution(
        nrows, ncols, fraction, method
    )

    save_bitstring_distribution(distribution, "distribution.json")


def get_thermal_states_target_distribution(**kwargs):
    distribution = _get_thermal_states_target_distribution(**kwargs)

    save_bitstring_distribution(distribution, "distribution.json")