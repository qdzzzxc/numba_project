import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def plot_multiple_bars(data: dict[str, tuple[list, float, float]], title: str, y_title: str, ymax: float = None, x_multiplier: float = 0.35):
    labels = list(data.keys())
    means = []
    stds = []

    for v in data.values():
        means.append(v.average)
        stds.append(v.stdev)

    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(labels))
    x = np.arange(len(labels)) * x_multiplier
    width = 0.3

    ax.bar(x, means, width, label='Mean', yerr=stds, capsize=5, color='blue', alpha=0.7, zorder=10)
    ax.bar(x, means + stds, width, label='Mean + Std', color='red', alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_title(title)
    #ax.legend()

    max_value = max(means + stds)
    step = max_value / 10

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='center')
    ax.set_yticks(np.arange(0, max_value + step, step))
    ax.grid(alpha=0.3, linestyle=':')
    
    ax.set_xlabel('Метод')
    ax.set_ylabel(y_title)
    if ymax:
        ax.set_ylim(0, ymax)
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    plt.tight_layout()
    plt.show()