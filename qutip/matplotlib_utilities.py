"""
This module contains utility functions that enhance Matplotlib
in one way or another.
"""

__all__ = ['wigner_cmap', 'MidpointNorm', 'complex_phase_cmap']

import numpy as np

try:
    import matplotlib as mpl
    from matplotlib import cm
    from matplotlib.colors import (Normalize, ColorConverter)
except:
    class Normalize(object):
        def __init__(self, vmin=None, vmax=None, clip=False):
            pass


def wigner_cmap(W, levels=1024, shift=0, max_color='#09224F',
                mid_color='#FFFFFF', min_color='#530017',
                neg_color='#FF97D4', invert=False):
    """A custom colormap that emphasizes negative values by creating a
    nonlinear colormap.

    Parameters
    ----------
    W : array
        Wigner function array, or any array.
    levels : int, default: 1024
        Number of color levels to create.
    shift : float, default: 0
        Shifts the value at which Wigner elements are emphasized.
        This parameter should typically be negative and small (i.e -1e-5).
    max_color : str, default: '#09224F'
        String for color corresponding to maximum value of data.  Accepts
        any string format compatible with the Matplotlib.colors.ColorConverter.
    mid_color : str, default: '#FFFFFF'
        Color corresponding to zero values.  Accepts any string format
        compatible with the Matplotlib.colors.ColorConverter.
    min_color : str, default: '#530017'
        Color corresponding to minimum data values.  Accepts any string format
        compatible with the Matplotlib.colors.ColorConverter.
    neg_color : str, default: '#FF97D4'
        Color that starts highlighting negative values.  Accepts any string
        format compatible with the Matplotlib.colors.ColorConverter.
    invert : bool, default: False
        Invert the color scheme for negative values so that smaller negative
        values have darker color.

    Returns
    -------
    Returns a Matplotlib colormap instance for use in plotting.

    Notes
    -----
    The 'shift' parameter allows you to vary where the colormap begins
    to highlight negative colors. This is beneficial in cases where there
    are small negative Wigner elements due to numerical round-off and/or
    truncation.

    """
    cc = ColorConverter()
    max_color = np.array(cc.to_rgba(max_color), dtype=float)
    mid_color = np.array(cc.to_rgba(mid_color), dtype=float)
    if invert:
        min_color = np.array(cc.to_rgba(neg_color), dtype=float)
        neg_color = np.array(cc.to_rgba(min_color), dtype=float)
    else:
        min_color = np.array(cc.to_rgba(min_color), dtype=float)
        neg_color = np.array(cc.to_rgba(neg_color), dtype=float)
    # get min and max values from Wigner function
    bounds = [W.min(), W.max()]
    # create empty array for RGBA colors
    adjust_RGBA = np.hstack((np.zeros((levels, 3)), np.ones((levels, 1))))
    zero_pos = int(np.round(levels * np.abs(shift - bounds[0])
                        / (bounds[1] - bounds[0])))
    num_pos = levels - zero_pos
    num_neg = zero_pos - 1
    # set zero values to mid_color
    adjust_RGBA[zero_pos] = mid_color
    # interpolate colors
    for k in range(0, levels):
        if k < zero_pos:
            interp = k / (num_neg + 1.0)
            adjust_RGBA[k][0:3] = (1.0 - interp) * \
                min_color[0:3] + interp * neg_color[0:3]
        elif k > zero_pos:
            interp = (k - zero_pos) / (num_pos + 1.0)
            adjust_RGBA[k][0:3] = (1.0 - interp) * \
                mid_color[0:3] + interp * max_color[0:3]
    # create colormap
    wig_cmap = mpl.colors.LinearSegmentedColormap.from_list('wigner_cmap',
                                                            adjust_RGBA,
                                                            N=levels)
    return wig_cmap


def complex_phase_cmap():
    """
    Create a cyclic colormap for representing the phase of complex variables

    Returns
    -------
    cmap :
        A matplotlib linear segmented colormap.
    """
    cdict = {'blue': ((0.00, 0.0, 0.0),
                      (0.25, 0.0, 0.0),
                      (0.50, 1.0, 1.0),
                      (0.75, 1.0, 1.0),
                      (1.00, 0.0, 0.0)),
             'green': ((0.00, 0.0, 0.0),
                       (0.25, 1.0, 1.0),
                       (0.50, 0.0, 0.0),
                       (0.75, 1.0, 1.0),
                       (1.00, 0.0, 0.0)),
             'red': ((0.00, 1.0, 1.0),
                     (0.25, 0.5, 0.5),
                     (0.50, 0.0, 0.0),
                     (0.75, 0.0, 0.0),
                     (1.00, 1.0, 1.0))}

    cmap = mpl.colors.LinearSegmentedColormap('phase_colormap', cdict, 256)

    return cmap


class MidpointNorm(Normalize):
    """Normalization for a colormap centered about a given midpoint.

    Parameters
    ----------
    midpoint : float (optional, default=0)
        Midpoint about which colormap is centered.
    vmin: float (optional)
        Minimal value for colormap. Calculated from data by default.
    vmax: float (optional)
        Maximal value for colormap. Calculated from data by default.

    Returns
    -------
    Returns a Matplotlib colormap normalization that can be used
    with any colormap.

    """

    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
