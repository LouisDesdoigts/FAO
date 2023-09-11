import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import jax.numpy as np


def add_cbar(
    fig,
    ax,
    im,
    label=None,
    location="right",
    size="5%",
    pad=0.1,
    orientation="vertical",
):
    """Adds a colorbar to a figure

    Params:
        fig: matplotlib figure
        ax: matplotlib axes
        location: str, location of the colorbar
        size: float, size of the colorbar
        pad: float, padding of the colorbar
        orientation: str, orientation of the colorbar

    Returns:
        fig: matplotlib figure
        ax: matplotlib axes"""
    if label is None:
        label = ""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size=size, pad=pad)
    fig.colorbar(im, cax=cax, orientation=orientation, label=label)
    return fig


def plot_grid_2d(rows, cols, fsx=5, fsy=4, nested=False):
    """Returns a figure and axes arranged in a 2d grid

    Params:
        rows: int, number of rows
        cols: int, number of columns
        fsx: float, individual figure size in x
        fsy: float, individual figure size in y
        nested: bool, whether to return a nested list of axes or a flat list

    Returns:
        fig: matplotlib figure
        axes: list of matplotlib axes arranged in a 2d grid"""
    fig = plt.figure(figsize=(fsx * cols, fsy * rows))
    axes = []
    for j in range(rows):
        axes_tmp = []
        for i in range(cols):
            ax = plt.subplot(rows, cols, i + j * cols + 1)

            if nested:
                axes_tmp.append(ax)
            else:
                axes.append(ax)
        if nested:
            axes.append(axes_tmp)
    return fig, axes


def compare_layers(initial, final):
    plt.figure(figsize=(20, 4))
    plt.subplot(1, 4, 1)
    plt.title("Initial Weights")
    plt.imshow(initial.weight)
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.title("Final Weights")
    plt.imshow(final.weight)
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.title("Weight Difference")
    plt.imshow(initial.weight - final.weight)
    plt.colorbar()

    plt.subplot(1, 4, 4)
    xs = np.arange(len(initial.bias))
    plt.scatter(xs, initial.bias, label="Initial")
    plt.scatter(xs, final.bias, label="Final")
    plt.legend()

    plt.tight_layout()
    plt.show()
