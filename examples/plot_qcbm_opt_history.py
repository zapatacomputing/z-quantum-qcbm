"""Plot the VQE binding energy curve of a diatomic molecule from a Quantum
Engine workflow result JSON."""

import json
import matplotlib
# matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

def get_ordered_list_of_bitstrings(num_qubits):
    bitstrings = []
    for i in range(2**num_qubits):
        bitstring = "{0:b}".format(i)
        while len(bitstring) < num_qubits:
            bitstring = "0" + bitstring
        bitstrings.append(bitstring)
    return bitstrings

# Insert the path to your JSON file here
with open('./examples/qcbm-example.json') as f:
    data = json.load(f)

# Extract lists of energies, bond lengths, and basis sets.
distances = []
minimum_distances = []
bistring_distributions = []

current_minimum = 100000
for step_id in data:
    step = data[step_id]
    if step["class"] == "optimize-variational-qcbm-circuit":
        ordered_bitstrings = get_ordered_list_of_bitstrings(int(step["inputParam:n-qubits"]))
        
        for evaluation in step["optimization-results"]["history"]:
            distances.append(evaluation["value"])
            current_minimum = min(current_minimum, evaluation["value"])
            minimum_distances.append(current_minimum)

            bitstring_dist = []
            for key in ordered_bitstrings:
                try:
                    bitstring_dist.append(evaluation["bitstring_distribution"][key])
                except:
                    bitstring_dist.append(0)
            bistring_distributions.append(bitstring_dist)
            

# Plot the optimization process
fig, ax = plt.subplots(nrows = 2, ncols=1, figsize=(16,8))

evals = []
plotted_distances = []
plotted_min_distances = []
line_widths = []

def animate(i):
    evals.append(i)
    plotted_distances.append(distances[i])
    plotted_min_distances.append(minimum_distances[i])
    line_widths.append(1)
    ax[0].clear()
    ax[0].set(xlabel='Evaluation Index', ylabel='Distribution Distance (clipped log-likelihood)')
    ax[0].set_ylim([1.5, 3.5])
    ax[0].scatter(evals, plotted_distances, color="green", linewidths=line_widths, marker=".")
    ax[0].plot(evals, plotted_min_distances, color="purple", linewidth=2)

    ax[1].clear()
    ax[1].set(xlabel='Bitstring', ylabel='Measured Probability')
    ax[1].set_ylim([0, .25])
    ax[1].bar(ordered_bitstrings, bistring_distributions[i], facecolor='green')
    return (ax[0], ax[1])

anim = FuncAnimation(fig, animate, frames=len(bistring_distributions), interval=1)

# # Set up formatting for the movie files
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
# anim.save('qcbm_opt.mp4', writer=writer)

plt.show()