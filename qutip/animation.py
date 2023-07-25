"""
Functions for creating animations of results of quantum dynamics simulations,
visualizations of quantum states and processes.
"""

__all__ = ['anim_wigner']

import warnings
import itertools as it
import numpy as np
from numpy import pi, array, sin, cos, angle, log2

from packaging.version import parse as parse_version

from . import (
    Qobj, isket, ket2dm, tensor, vector_to_operator, to_super, settings
)
from .core.dimensions import flatten
from .core.superop_reps import _to_superpauli, isqubitdims
from .wigner import wigner
from .matplotlib_utilities import complex_phase_cmap

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    # Define a custom _axes3D function based on the matplotlib version.
    # The auto_add_to_figure keyword is new for matplotlib>=3.4.
    if parse_version(mpl.__version__) >= parse_version('3.4'):
        def _axes3D(fig, *args, **kwargs):
            ax = Axes3D(fig, *args, auto_add_to_figure=False, **kwargs)
            return fig.add_axes(ax)
    else:
        def _axes3D(*args, **kwargs):
            return Axes3D(*args, **kwargs)
except:
    pass

import os
from functools import partial
from IPython.display import HTML
from base64 import b64encode
import matplotlib.animation as animation
from qutip import plot_wigner
import matplotlib.pyplot as plt
import matplotlib as mpl
from . import matrix_histogram, plot_wigner


def _same_fig(fig, func_options):
    if fig is None:
        if 'fig' not in func_options.keys():
            fig = plt.figure()
            func_options['fig'] = fig
        else:
            fig = func_options['fig']
    else:
        if 'fig' not in func_options.keys():
            func_options['fig'] = fig
        else:
            error = "fig and func_options['fig'] must be the same"
            if fig != func_options['fig']:
                raise ValueError(error)

    return fig, func_options


def _is_options(options, arg_name):
    if options is None:
        options = dict()

    if not isinstance(options, dict):
        raise TypeError(str(arg_name) + " must be a dict")

    return options


def _delete_axes(fig):
    for ax in fig.axes:
        fig.delaxes(ax)


def _default_setup(ax, frame):
    return ax


def make_html_video(ani):
    file_name = 'animation_for_video.mp4'
    ani.save(file_name, fps=10)
    video = open(file_name, "rb").read()
    os.remove(file_name)
    video_encoded = b64encode(video).decode("ascii")
    video_tag = '<video controls src="data:video/x-m4v;base64,{0}">'.format(
        video_encoded)

    return ani, HTML(video_tag)


def anim_wigner(rhos, xvec=None, yvec=None, method='clenshaw',
                projection='2d', *, cmap=None, colorbar=False,
                fig=None, ax=None):
    """
    Plot the the Wigner function for a density matrix (or ket) that describes
    an oscillator mode.

    Parameters
    ----------
    rho : :class:`qutip.Qobj`
        The density matrix (or ket) of the state to visualize.

    xvec : array_like, optional
        x-coordinates at which to calculate the Wigner function.

    yvec : array_like, optional
        y-coordinates at which to calculate the Wigner function.  Does not
        apply to the 'fft' method.

    method : string {'clenshaw', 'iterative', 'laguerre', 'fft'},
        default='clenshaw'
        The method used for calculating the wigner function. See the
        documentation for qutip.wigner for details.

    projection: string {'2d', '3d'}, default='2d'
        Specify whether the Wigner function is to be plotted as a
        contour graph ('2d') or surface plot ('3d').

    cmap : a matplotlib cmap instance, optional
        The colormap.

    colorbar : bool, default=False
        Whether (True) or not (False) a colorbar should be attached to the
        Wigner function graph.

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The axes context in which the plot will be drawn.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.
    """

    if projection not in ('2d', '3d'):
        raise ValueError('Unexpected value of projection keyword argument')
    if projection == '2d':
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel(r'$\rm{Re}(\alpha)$', fontsize=12)
    ax.set_ylabel(r'$\rm{Im}(\alpha)$', fontsize=12)
    ax.set_title("Wigner function", fontsize=12)
    wlim = -1
    Ws = list()
    artist_list = list()

    for rho in rhos:
        if isket(rho):
            rho = ket2dm(rho)

        if xvec is None:
            xvec = np.linspace(-7.5, 7.5, 200)
        if yvec is None:
            yvec = np.linspace(-7.5, 7.5, 200)

        W0 = wigner(rho, xvec, yvec, method=method)

        W, yvec = W0 if isinstance(W0, tuple) else (W0, yvec)
        Ws.append(W)

        wlim = max(abs(W).max(), wlim)

    norm = mpl.colors.Normalize(-wlim, wlim)
    if cmap is None:
        cmap = mpl.cm.RdBu

    for W in Ws:
        if projection == '2d':
            cf = ax.contourf(xvec, yvec, W, 100, norm=norm, cmap=cmap).collections
        else:
            X, Y = np.meshgrid(xvec, yvec)
            cf = [ax.plot_surface(X, Y, W, rstride=5, cstride=5, linewidth=0.5,
                                norm=norm, cmap=cmap)]
        artist_list.append(cf)


    if colorbar:
        cax, kw = mpl.colorbar.make_axes(ax, pad=.1)
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

    ani = animation.ArtistAnimation(fig, artist_list, interval=50, blit=True,
                                repeat_delay=1000)
    return make_html_video(ani)
