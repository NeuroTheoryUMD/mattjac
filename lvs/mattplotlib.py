# utilities to make plotting data simpler and easier

import matplotlib.pyplot as plt
import NDNT.utils as utils # some other utilities
import math


# make new figure for simplicity
def fig():
    # close all figures if we have too many to avoid memory leak
    if plt.gcf().number > 20:
        plt.close("all")
    # make a new figure
    return plt.figure()


# make a bunch of subplots of the provided vectors, and their names
def plots(*args, title=None):
    # show the legend if the labels are provided
    show_legend = True if type(args[0]) is tuple else False
    for i, arg in enumerate(args):
        plt.subplot(len(args), 1, i+1)
        # show the title if provided, but it has to be on top of the first subplot
        if i == 0 and title is not None:
            plt.title(title)
        if show_legend:
            plt.plot(arg[0], label=arg[1])
            plt.legend()
        else:
            plt.plot(arg)
    plt.show()


# turn a flat list of things into a grid of things
def list_to_grid(objects, width):
    height = math.ceil(len(objects) / width)
    grid = []
    for i in range(height):
        grid.append([])
        for j in range(width):
            idx = j + width*i
            if idx < len(objects):
                grid[i].append(objects[idx])
            else:
                grid[i].append(None)
    return grid, height


def plot_glm_filters(glm, nc):
    ws = glm.get_weights()
    utils.ss(5,6)

    for cc in range(nc):
        plt.subplot(5,6, cc+1)
        utils.imagesc(ws[:,:,cc])
        plt.colorbar()
    plt.show()




