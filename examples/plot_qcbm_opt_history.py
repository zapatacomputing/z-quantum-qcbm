import json
import matplotlib

# matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.cbook import get_sample_data
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import entropy


def get_ordered_list_of_bitstrings(num_qubits):
    bitstrings = []
    for i in range(2 ** num_qubits):
        bitstring = "{0:b}".format(i)
        while len(bitstring) < num_qubits:
            bitstring = "0" + bitstring
        bitstrings.append(bitstring)
    return bitstrings


# Insert the path to your JSON file here
with open("workflow_result.json") as f:
    data = json.load(f)

# Extract target/measured bitstring distribution and distance measure values.
distances = []
minimum_distances = []
bitstring_distributions = []

current_minimum = 100000
number_of_qubits = 4
for step_id in data:
    step = data[step_id]
    # if step["stepName"] == "get-initial-parameters":
    #     number_of_qubits = int(
    #         eval(step["inputParam:ansatz-specs"])["number_of_qubits"]
    #     )
    ordered_bitstrings = get_ordered_list_of_bitstrings(number_of_qubits)
    if step["stepName"] == "get-bars-and-stripes-distribution":
        target_distribution = []
        for key in ordered_bitstrings:
            try:
                target_distribution.append(
                    step["distribution"]["bitstring_distribution"][key]
                )
            except:
                target_distribution.append(0)
        exact_distance_value = entropy(target_distribution)
        print(exact_distance_value)
    elif step["stepName"] == "optimize-circuit":
        for evaluation in step["qcbm-optimization-results"]["history"]:
            distances.append(evaluation["value"]["value"])
            current_minimum = min(current_minimum, evaluation["value"]["value"])
            minimum_distances.append(current_minimum)

            bitstring_dist = []
            for key in ordered_bitstrings:
                try:
                    bitstring_dist.append(
                        evaluation["artifacts"]["bitstring_distribution"][key]
                    )
                except:
                    bitstring_dist.append(0)
            bitstring_distributions.append(bitstring_dist)

fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(nrows=8, ncols=3, width_ratios=[16, 1, 1])
ax1 = fig.add_subplot(gs[:4, 0])
ax2 = fig.add_subplot(gs[5:, 0])
axs = [fig.add_subplot(gs[i, 1]) for i in range(8)] + [
    fig.add_subplot(gs[i, 2]) for i in range(8)
]

evals = []
plotted_distances = []
plotted_min_distances = []
line_widths = []

images = [
    np.array([0, 0, 0, 0]),
    np.array([0, 0, 0, 1]),
    np.array([0, 0, 1, 0]),
    np.array([0, 0, 1, 1]),
    np.array([0, 1, 0, 0]),
    np.array([0, 1, 0, 1]),
    np.array([0, 1, 1, 0]),
    np.array([0, 1, 1, 1]),
    np.array([1, 0, 0, 0]),
    np.array([1, 0, 0, 1]),
    np.array([1, 0, 1, 0]),
    np.array([1, 0, 1, 1]),
    np.array([1, 1, 0, 0]),
    np.array([1, 1, 0, 1]),
    np.array([1, 1, 1, 0]),
    np.array([1, 1, 1, 1]),
]


def animate(i):
    evals.append(i)
    plotted_distances.append(distances[i])
    plotted_min_distances.append(minimum_distances[i])
    line_widths.append(1)
    ax1.clear()
    ax1.set(
        xlabel="Evaluation Index",
        ylabel="Clipped negative log-likelihood cost function",
    )
    ax1.set_ylim([exact_distance_value - 0.1, exact_distance_value + 1.5])
    ax1.scatter(
        evals, plotted_distances, color="green", linewidths=line_widths, marker="."
    )
    ax1.hlines(
        y=exact_distance_value,
        xmin=0,
        xmax=evals[-1],
        color="darkgreen",
        label="expected",
        alpha=0.8,
        linestyle="--",
    )
    ax1.legend(loc="upper right")
    ax1.plot(evals, plotted_min_distances, color="purple", linewidth=2)

    ax2.clear()
    ax2.set(xlabel="Bitstring", ylabel="Measured Probability")
    ax2.set_ylim([0, np.max(target_distribution) + 0.05])
    
    x_locations = np.arange(16)

    # Create the bars at the x locations with the height of the proportions for each bitstring

    ax2.bar(x_locations, bitstring_distributions[i], facecolor="green")
    ax2.bar(
        x_locations,
        target_distribution,
        facecolor="darkgreen",
        alpha=0.3,
        label="target",
    )

    # set the tick locations as the center of the bar, then set the label to be the bitstring
    ax2.set_xticks([x+0.4 for x in x_locations])
    ax2.set_xticklabels(ordered_bitstrings)
    ax2.legend(loc="upper right")
    
    if distances[i] == minimum_distances[i]:
        normalized_distribution = np.array(bitstring_distributions[i]) / max(
            bitstring_distributions[i]
        )
        for j in range(len(ordered_bitstrings)):
            axs[j].clear()
            axs[j].set_xticks(np.arange(-0.5, 2, 1), minor=True)
            axs[j].set_yticks(np.arange(-0.5, 2, 1), minor=True)
            axs[j].tick_params(axis="x", colors=(0, 0, 0, 0))
            axs[j].tick_params(axis="y", colors=(0, 0, 0, 0))

            axs[j].grid(which="minor", color="k", linestyle="-", linewidth=2)
            fading_factor = normalized_distribution[j]
            axs[j].imshow(
                (images[j].reshape((2, 2))),
                alpha=fading_factor,
                vmin=0,
                vmax=1,
                cmap="PiYG",
            )

    return tuple([ax1, ax2] + axs)


anim = FuncAnimation(fig, animate, frames=700, interval=1, repeat=False)

# # Set up formatting for the movie files
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
# anim.save('qcbm_opt_700_iterations.mp4', writer=writer)

plt.show()
