#!/usr/bin/env python
import matplotlib as mpl
from matplotlib import pyplot as pl
import numpy as np


def discrete_color_norm(vmin, vmax, bins, log=False, cmap=pl.cm.jet,
                        return_cmap=False):
    # define the colormap
    # cmap = pl.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    # cmaplist[0] = (.5,.5,.5,1.0)
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    if log:
        bounds = np.logspace(np.log10(vmin), np.log10(vmax), bins+1, base=10.)
    else:
        bounds = np.linspace(vmin, vmax, bins+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    if return_cmap:
        return norm, cmap
    else:
        return norm


def implot(grid, title=None, vmin=None, vmax=None, extent=None):
    pl.figure()
    pl.imshow(grid, vmin=vmin, vmax=vmax, extent=extent)
    if title:
        pl.title(title)
    pl.colorbar()


def impl(grid_name, vmin=None, vmax=None, extent=None):
    if extent:
        if type(extent) not in (list, tuple):
            extent = (0, extent, 0, extent)
    implot(eval(grid_name), grid_name, vmin, vmax, extent)


def add_shaded_region(line, ax, deviation, deviation_low=None, alpha=0.5):
    """
    line: matplotlib line object, which is returned from plot function.
    If deviation_low is None: use deviation for both upper and lower boundary.
    """
    x = line.get_xdata()
    y = line.get_ydata()
    if deviation_low is None:
        deviation_low = deviation
    shade = ax.fill_between(x, y - deviation_low, y + deviation, alpha=alpha,
                            color=line.get_color())
    return shade
