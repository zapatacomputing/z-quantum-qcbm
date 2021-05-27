from zquantum.qcbm.ansatz import QCBMAnsatz
from matplotlib import pyplot as plt
import numpy as np
from scipy import sparse
import time

nlayers = 2
nqubits = 6
connectivity = np.array(
    [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)

start = time.time()
ansatz = QCBMAnsatz(nlayers, nqubits, "graph", adjacency_matrix=connectivity)

end = time.time()
print("Time: ", end - start)
print(ansatz.parametrized_circuit)
