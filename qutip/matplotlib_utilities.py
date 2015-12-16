# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

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
    levels : int
        Number of color levels to create.
    shift : float
        Shifts the value at which Wigner elements are emphasized.
        This parameter should typically be negative and small (i.e -1e-5).
    max_color : str
        String for color corresponding to maximum value of data.  Accepts
        any string format compatible with the Matplotlib.colors.ColorConverter.
    mid_color : str
        Color corresponding to zero values.  Accepts any string format
        compatible with the Matplotlib.colors.ColorConverter.
    min_color : str
        Color corresponding to minimum data values.  Accepts any string format
        compatible with the Matplotlib.colors.ColorConverter.
    neg_color : str
        Color that starts highlighting negative values.  Accepts any string
        format compatible with the Matplotlib.colors.ColorConverter.
    invert : bool
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
    zero_pos = np.round(levels * np.abs(shift - bounds[0])
                        / (bounds[1] - bounds[0]))
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

# Added 16 December 2015
# by Alexander Pitchford
def plot_ctrl_pulse(time, amps, ax=None, **plot_kwargs):
    """
    Plot the control pulse amplitudes on Matplotlib axes. 
    Specifically for control pulses produced by control.pulseoptim
    functions.
    
    If ax is passed, then these axes will be used, and it will be left
    to the user to 'show' the plot.
    Otherwise a fresh set of axes will be created and the plot shown - note
    this will block the code.
    
    A simple line plot is produced, with two points per timeslot to
    make clear that the pulse is constant within the timeslot
    
    Parameters
    ----------
    amps : ndarray of float
        These are the pulse amplitudes to be plotted
        If the array is 2d, for instance of OptimResult.final_amps is passed
        when there are muliple controls, then it is assumed that each column
        is a pulse and each will be plotted. If specific parameters are to
        be used for each (label for example), then this function should
        be separately for each pulse (column)
        If the array is 1d, then it is assumed to be a single pulse
        
    time : array[num_tslots+1] of float
        Time of the start of each timeslot
        with the final value being the total evolution time
        OptimResult.time can be used for this
        
    ax : matplotlib.AxesSubplot
        Axes upon which the plot will be made
        
    plot_kwargs : kwargs (dict)
        These will be past to the ax.plot function call
    """
    

    show = False
    if ax is None:
        plt = mpl.pyplot
        fig1 = plt.figure()
        ax = fig1.add_subplot(1, 1, 1)
        ax.set_title("Control Pulse")
        ax.set_xlabel("Time")
        ax.set_ylabel("Control amplitude")
        show = True
        
    try:
        if len(time.shape) != 1:
            raise ValueError("time is expected to be a 1d array")
    except:
        raise ValueError("time is expected to be a 1d array")
    
    n_ts = len(time) - 1
    n_pts = 2*n_ts
    
    x = np.zeros([n_pts])
    x[::2] = time[:-1]
    x[1::2] = time[1:]
    y = np.zeros([n_pts])
    try:
        if len(amps.shape) == 1:
            n_pulse = 1
        else:
            n_pulse = amps.shape[1]
            if n_pulse == 1:
                amps = amps[:, 0]
    except:
        raise ValueError(
                "Unable to plot the amps, suspect incorrect array shape")
    
    if n_pulse == 1:
        y[::2] = amps
        y[1::2] = amps
        ax.plot(x, y, **plot_kwargs)
    else:
        for j in range(n_pulse):
            y[::2] = amps[:, j]
            y[1::2] = amps[:, j]
            ax.plot(x, y, **plot_kwargs)
        
    if show:
        plt.show()
        