"""
Functions for creating animations of results of quantum dynamics simulations,
visualizations of quantum states and processes.
"""

__all__ = ['anim_wigner_sphere', 'anim_hinton', 'anim_sphereplot',
           'anim_matrix_histogram', 'anim_fock_distribution',
           'anim_wigner', 'anim_spin_distribution', 'anim_qubism',
           'plot_schmidt']

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
from .visualization import *


def _is_options(options, keyward):
    if options is None:
        options = dict()

    if not isinstance(options, dict):
        raise ValueError(str(keyward) + ' must be a dict')

    return options


def make_html_video(ani, save_options=None):
    save_options = _is_options(save_options, 'save_options')
    if 'name' not in save_options.keys():
        save_options['name'] = 'animation'
    if 'writer' not in save_options.keys():
        save_options['writer'] = None
    if 'codec' not in save_options.keys():
        save_options['codec'] = None

    ani.save(save_options['name'] + '.mp4', fps=10,
             writer=save_options['writer'], codec=save_options['codec'])
    video = open(save_options['name'] + '.mp4', "rb").read()
    video_encoded = b64encode(video).decode("ascii")
    video_tag = '<video controls src="data:video/x-m4v;base64,{0}">'.format(
        video_encoded)

    return ani, HTML(video_tag)


def anim_wigner_sphere(wigners, reflections=False, *, cmap=None,
                       colorbar=True, fig=None, ax=None,
                       save_options=None):
    """Plots a coloured Bloch sphere.

    Parameters
    ----------
    wigners : list of wigner functions
        The wigner transformation at `steps` different theta and phi.

    reflections : bool, default=False
        If the reflections of the sphere should be plotted as well.

    cmap : a matplotlib colormap instance, optional
        Color map to use when plotting.

    colorbar : bool, default=True
        Whether (True) or not (False) a colorbar should be attached.

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The ax context in which the plot will be drawn.

    save_options : dict, optional
        A dictionary containing options to save the animation.

        'name' : str, default='animation'
            The output filename, e.g., :file:`mymovie.mp4`.

        'writer' : `MovieWriter` or str, optional
            A `MovieWriter` instance to use or a key that identifies a
            class to use, such as 'ffmpeg'.

        'codec' : str, optional
            The video codec to use.  Not all codecs are supported by a given
            `MovieWriter`.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.

    Notes
    -----
    Special thanks to Russell P Rundle for writing this function.
    """

    fig, ax = _is_fig_and_ax(fig, ax, projection='3d')

    if cmap is None:
        cmap = _diverging_cmap()

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    wigner_max = 0
    for wigner in wigners:
        wigner_max = max(np.real(np.amax(np.abs(wigner))), wigner_max)

    artist_list = list()
    for wigner in wigners:
        steps = len(wigner)

        theta = np.linspace(0, np.pi, steps)
        phi = np.linspace(0, 2 * np.pi, steps)
        x = np.outer(np.sin(theta), np.cos(phi))
        y = np.outer(np.sin(theta), np.sin(phi))
        z = np.outer(np.cos(theta), np.ones(steps))
        wigner = np.real(wigner)

        wigner_c1 = cmap((wigner + wigner_max) / (2 * wigner_max))

        artist = list()
        # Plot coloured Bloch sphere:
        artist.append(ax.plot_surface(x, y, z, facecolors=wigner_c1,
                                      vmin=-wigner_max, vmax=wigner_max,
                                      rcount=steps, ccount=steps, linewidth=0,
                                      zorder=0.5, antialiased=None))

        if reflections:
            wigner_c2 = cmap((wigner[0:steps, 0:steps]+wigner_max) /
                             (2*wigner_max))  # bottom
            wigner_c3 = cmap((wigner[0:steps, 0:steps]+wigner_max) /
                             (2*wigner_max))  # side
            wigner_c4 = cmap((wigner[0:steps, 0:steps]+wigner_max) /
                             (2*wigner_max))  # back

            # Plot bottom reflection:
            artist.append(ax.plot_surface(x[0:steps, 0:steps],
                                          y[0:steps, 0:steps],
                                          -1.5*np.ones((steps, steps)),
                                          facecolors=wigner_c2,
                                          vmin=-wigner_max, vmax=wigner_max,
                                          rcount=steps/2, ccount=steps/2,
                                          linewidth=0, zorder=0.5,
                                          antialiased=False))

            # Plot side reflection:
            artist.append(ax.plot_surface(-1.5*np.ones((steps, steps)),
                                          y[0:steps, 0:steps],
                                          z[0:steps, 0:steps],
                                          facecolors=wigner_c3,
                                          vmin=-wigner_max, vmax=wigner_max,
                                          rcount=steps/2, ccount=steps/2,
                                          linewidth=0, zorder=0.5,
                                          antialiased=False))

            # Plot back reflection:
            artist.append(ax.plot_surface(x[0:steps, 0:steps],
                                          1.5*np.ones((steps, steps)),
                                          z[0:steps, 0:steps],
                                          facecolors=wigner_c4,
                                          vmin=-wigner_max, vmax=wigner_max,
                                          rcount=steps/2, ccount=steps/2,
                                          linewidth=0, zorder=0.5,
                                          antialiased=False))
        artist_list.append(artist)

    # Create colourbar:
    if colorbar:
        norm = mpl.colors.Normalize(-wigner_max, wigner_max)
        cax, kw = mpl.colorbar.make_axes(ax, shrink=0.75, pad=.1)
        mpl.colorbar.ColorbarBase(cax, norm=norm, cmap=cmap)

    ani = animation.ArtistAnimation(fig, artist_list, interval=50,
                                    blit=True, repeat_delay=1000)
    plt.close()

    ani, html_video = make_html_video(ani, save_options)

    return fig, ani, html_video


# Adopted from the SciPy Cookbook.
def anim_hinton(rhos, x_basis=None, y_basis=None, color_style="scaled",
                label_top=True, *, cmap=None, colorbar=True, fig=None,
                ax=None, save_options=None):
    """Draws a Hinton diagram to visualize a density matrix or superoperator.

    Parameters
    ----------
    rhos : list of qobj
        List of input density matrices or superoperators.

        .. note::

            Hinton plots of superoperators are currently only
            supported for qubits.

    x_basis : list of strings, optional
        list of x ticklabels to represent x basis of the input.

    y_basis : list of strings, optional
        list of y ticklabels to represent y basis of the input.

    color_style : string, default="scaled"

        Determines how colors are assigned to each square:

        -  If set to ``"scaled"`` (default), each color is chosen by
           passing the absolute value of the corresponding matrix
           element into `cmap` with the sign of the real part.
        -  If set to ``"threshold"``, each square is plotted as
           the maximum of `cmap` for the positive real part and as
           the minimum for the negative part of the matrix element;
           note that this generalizes `"threshold"` to complex numbers.
        -  If set to ``"phase"``, each color is chosen according to
           the angle of the corresponding matrix element.

    label_top : bool, default=True
        If True, x ticklabels will be placed on top, otherwise
        they will appear below the plot.

    cmap : a matplotlib colormap instance, optional
        Color map to use when plotting.

    colorbar : bool, default=True
        Whether (True) or not (False) a colorbar should be attached.

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The ax context in which the plot will be drawn.

    save_options : dict, optional
        A dictionary containing options to save the animation.

        'name' : str, default='animation'
            The output filename, e.g., :file:`mymovie.mp4`.

        'writer' : `MovieWriter` or str, optional
            A `MovieWriter` instance to use or a key that identifies a
            class to use, such as 'ffmpeg'.

        'codec' : str, optional
            The video codec to use.  Not all codecs are supported by a given
            `MovieWriter`.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.

    Raises
    ------
    ValueError
        Input argument is not a quantum object.

    Examples
    --------
    >>> import qutip
    >>>
    >>> dm = qutip.rand_dm(4)
    >>> fig, ax = qutip.hinton(dm)
    >>> fig.show()
    >>>
    >>> qutip.settings.colorblind_safe = True
    >>> fig, ax = qutip.hinton(dm, color_style="threshold")
    >>> fig.show()
    >>> qutip.settings.colorblind_safe = False
    >>>
    >>> fig, ax = qutip.hinton(dm, color_style="phase")
    >>> fig.show()
    """

    fig, ax = _is_fig_and_ax(fig, ax)

    Ws = list()
    w_max = 0
    for rho in rhos:
        # Extract plotting data W from the input.
        if isinstance(rho, Qobj):
            if rho.isoper or rho.isoperket or rho.isoperbra:
                if rho.isoperket:
                    rho = vector_to_operator(rho)
                elif rho.isoperbra:
                    rho = vector_to_operator(rho.dag())
                W = rho.full()
                # Create default labels if none are given.
                if x_basis is None or y_basis is None:
                    labels = _cb_labels(rho.dims[0])
                    if x_basis is None:
                        x_basis = list(labels[0])
                    if y_basis is None:
                        y_basis = list(labels[1])

            elif rho.issuper:
                if not isqubitdims(rho.dims):
                    raise ValueError("Hinton plots of superoperators are "
                                     "currently only supported for qubits.")
                # Convert to a superoperator in the Pauli basis,
                # so that all the elements are real.
                sqobj = _to_superpauli(rho)
                nq = int(log2(sqobj.shape[0]) / 2)
                W = sqobj.full().T
                # Create default labels, too.
                if (x_basis is None) or (y_basis is None):
                    labels = list(map("".join, it.product("IXYZ", repeat=nq)))
                    if x_basis is None:
                        x_basis = labels
                    if y_basis is None:
                        y_basis = labels

            else:
                raise ValueError(
                    "Input quantum object must be "
                    "an operator or superoperator.")
        else:
            W = rho
        Ws.append(W)

        height, width = W.shape

        w_max = max(1.25 * max(abs(np.array(W)).flatten()), w_max)
        if w_max <= 0.0:
            w_max = 1.0

    # Set color_fn here.
    if color_style == "scaled":
        if cmap is None:
            cmap = _diverging_cmap()

        def color_fn(w):
            w = np.abs(w) * np.sign(np.real(w))
            return cmap(int((w + w_max) * 256 / (2 * w_max)))
    elif color_style == "threshold":
        if cmap is None:
            cmap = _diverging_cmap()

        def color_fn(w):
            w = np.real(w)
            return cmap(255 if w > 0 else 0)
    elif color_style == "phase":
        if cmap is None:
            cmap = _cyclic_cmap()

        def color_fn(w):
            return cmap(int(255 * (np.angle(w) / 2 / np.pi + 0.5)))
    else:
        raise ValueError(
            "Unknown color style {} for Hinton diagrams.".format(color_style)
        )

    artist_list = list()
    ax.fill(array([0, width, width, 0]), array([0, 0, height, height]),
            color=cmap(128))
    for W in Ws:
        artist = list()
        for x in range(width):
            for y in range(height):
                _x = x + 1
                _y = y + 1
                artist += _blob(_x - 0.5, height - _y + 0.5, W[y, x],
                                w_max, min(1, abs(W[y, x]) / w_max),
                                color_fn=color_fn, ax=ax)
        artist_list.append(artist)

    if colorbar:
        vmax = np.pi if color_style == "phase" else w_max
        norm = mpl.colors.Normalize(-vmax, vmax)
        cax, kw = mpl.colorbar.make_axes(ax, shrink=0.75, pad=.1)
        mpl.colorbar.ColorbarBase(cax, norm=norm, cmap=cmap)

    # axis
    if not (x_basis or y_basis):
        ax.axis('off')
    ax.axis('equal')
    ax.set_frame_on(False)

    # x axis
    xticks = 0.5 + np.arange(width)
    if x_basis:
        _set_ticklabels(ax, x_basis, xticks, 'x')
    if label_top:
        ax.xaxis.tick_top()

    # y axis
    yticks = 0.5 + np.arange(height)
    if y_basis:
        _set_ticklabels(ax, list(reversed(y_basis)), yticks, 'y')

    ani = animation.ArtistAnimation(fig, artist_list, interval=50,
                                    blit=True, repeat_delay=1000)
    plt.close()

    ani, html_video = make_html_video(ani, save_options)

    return fig, ani, html_video


def anim_sphereplot(theta, phi, V, *, cmap=None, colorbar=True,
                    fig=None, ax=None, save_options=None):
    """Plots a matrix of values on a sphere

    Parameters
    ----------
    theta : float
        Angle with respect to z-axis. Its range is between 0 and pi

    phi : float
        Angle in x-y plane. Its range is between 0 and 2*pi

    V : list
        List of data sets to be plotted

    cmap : a matplotlib colormap instance, optional
        Color map to use when plotting.

    colorbar : bool, default=True
        Whether (True) or not (False) a colorbar should be attached.

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The axes context in which the plot will be drawn.

    save_options : dict, optional
        A dictionary containing options to save the animation.

        'name' : str, default='animation'
            The output filename, e.g., :file:`mymovie.mp4`.

        'writer' : `MovieWriter` or str, optional
            A `MovieWriter` instance to use or a key that identifies a
            class to use, such as 'ffmpeg'.

        'codec' : str, optional
            The video codec to use.  Not all codecs are supported by a given
            `MovieWriter`.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.
    """
    fig, ax = _is_fig_and_ax(fig, ax, projection='3d')

    if cmap is None:
        cmap = _sequential_cmap()

    thetam, phim = np.meshgrid(theta, phi)
    xx = sin(thetam) * cos(phim)
    yy = sin(thetam) * sin(phim)
    zz = cos(thetam)
    min_ph = pi
    max_ph = -pi

    r_and_ph = list()
    for values in V:
        r = array(abs(values))
        ph = angle(values)
        min_ph = min(min_ph, ph.min())
        max_ph = max(max_ph, ph.max())
        r_and_ph.append((r, ph))
    # normalize color range based on phase angles in list ph
    norm = mpl.colors.Normalize(min_ph, max_ph)

    # plot with facecolors set to cm.jet colormap normalized to nrm
    artist_list = list()
    for r, ph in r_and_ph:
        artist = [ax.plot_surface(r * xx, r * yy, r * zz, rstride=1, cstride=1,
                                  facecolors=cmap(norm(ph)), linewidth=0,)]
        artist_list.append(artist)

    if colorbar:
        # create new axes on plot for colorbar and shrink it a bit.
        # pad shifts location of bar with repsect to the main plot
        cax, kw = mpl.colorbar.make_axes(ax, shrink=.66, pad=.05)

        # create new colorbar in axes cax with cmap and normalized to nrm like
        # our facecolors
        cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        # add our colorbar label
        cb1.set_label('Angle')

    ani = animation.ArtistAnimation(fig, artist_list, interval=50,
                                    blit=True, repeat_delay=1000)
    plt.close()

    ani, html_video = make_html_video(ani, save_options)

    return fig, ani, html_video


def anim_matrix_histogram(Ms, x_basis=None, y_basis=None, limits=None,
                          bar_style='real', color_limits=None,
                          color_style='real', options=None, *,
                          cmap=None, colorbar=True, fig=None, ax=None,
                          save_options=None):
    """
    Draw a histogram for the matrix M, with the given x and y labels and title.

    Parameters
    ----------
    Ms : list of matrices of Qobj
        list of matrices to visualize

    x_basis : list of strings, optional
        list of x ticklabels

    y_basis : list of strings, optional
        list of y ticklabels

    limits : list/array with two float numbers, optional
        The z-axis limits [min, max]

    bar_style : string, default="real"

        -  If set to ``"real"`` (default), each bar is plotted
           as the real part of the corresponding matrix element
        -  If set to ``"img"``, each bar is plotted
           as the imaginary part of the corresponding matrix element
        -  If set to ``"abs"``, each bar is plotted
           as the absolute value of the corresponding matrix element
        -  If set to ``"phase"`` (default), each bar is plotted
           as the angle of the corresponding matrix element

    color_limits : list/array with two float numbers, optional
        The limits of colorbar [min, max]

    color_style : string, default="real"
        Determines how colors are assigned to each square:

        -  If set to ``"real"`` (default), each color is chosen
           according to the real part of the corresponding matrix element.
        -  If set to ``"img"``, each color is chosen according to
           the imaginary part of the corresponding matrix element.
        -  If set to ``"abs"``, each color is chosen according to
           the absolute value of the corresponding matrix element.
        -  If set to ``"phase"``, each color is chosen according to
           the angle of the corresponding matrix element.

    cmap : a matplotlib colormap instance, optional
        Color map to use when plotting.

    colorbar : bool, default=True
        show colorbar

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The axes context in which the plot will be drawn.

    options : dict, optional
        A dictionary containing extra options for the plot.
        The names (keys) and values of the options are
        described below:

        'zticks' : list of numbers, optional
            A list of z-axis tick locations.

        'bars_spacing' : float, default=0.1
            spacing between bars.

        'bars_alpha' : float, default=1.
            transparency of bars, should be in range 0 - 1

        'bars_lw' : float, default=0.5
            linewidth of bars' edges.

        'bars_edgecolor' : color, default='k'
            The colors of the bars' edges.
            Examples: 'k', (0.1, 0.2, 0.5) or '#0f0f0f80'.

        'shade' : bool, default=True
            Whether to shade the dark sides of the bars (True) or not (False).
            The shading is relative to plot's source of light.

        'azim' : float, default=-35
            The azimuthal viewing angle.

        'elev' : float, default=35
            The elevation viewing angle.

        'stick' : bool, default=False
            Changes xlim and ylim in such a way that bars next to
            XZ and YZ planes will stick to those planes.
            This option has no effect if ``ax`` is passed as a parameter.

        'cbar_pad' : float, default=0.04
            The fraction of the original axes between the colorbar
            and the new image axes.
            (i.e. the padding between the 3D figure and the colorbar).

        'cbar_to_z' : bool, default=False
            Whether to set the color of maximum and minimum z-values to the
            maximum and minimum colors in the colorbar (True) or not (False).

        'threshold': float, optional
            Threshold for when bars of smaller height should be transparent. If
            not set, all bars are colored according to the color map.

    save_options : dict, optional
        A dictionary containing options to save the animation.

        'name' : str, default='animation'
            The output filename, e.g., :file:`mymovie.mp4`.

        'writer' : `MovieWriter` or str, optional
            A `MovieWriter` instance to use or a key that identifies a
            class to use, such as 'ffmpeg'.

        'codec' : str, optional
            The video codec to use.  Not all codecs are supported by a given
            `MovieWriter`.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.

    Raises
    ------
    ValueError
        Input argument is not valid.

    """

    # default options
    default_opts = {'zticks': None, 'bars_spacing': 0.2,
                    'bars_alpha': 1., 'bars_lw': 0.5, 'bars_edgecolor': 'k',
                    'shade': True, 'azim': -35, 'elev': 35, 'stick': False,
                    'cbar_pad': 0.04, 'cbar_to_z': False, 'threshold': None}

    # update default_opts from input options
    if options is None:
        options = dict()

    if isinstance(options, dict):
        # check if keys in options dict are valid
        if set(options) - set(default_opts):
            raise ValueError("invalid key(s) found in options: "
                             f"{', '.join(set(options) - set(default_opts))}")
        else:
            # updating default options
            default_opts.update(options)
            options = default_opts
    else:
        raise ValueError("options must be a dictionary")

    _bars_and_colors = list()
    for i in range(len(Ms)):
        M = Ms[i]
        if isinstance(M, Qobj):
            if x_basis is None:
                x_basis = list(_cb_labels([M.shape[0]])[0])
            if y_basis is None:
                y_basis = list(_cb_labels([M.shape[1]])[1])
            # extract matrix data from Qobj
            M = M.full()

        n = np.size(M)
        xpos, ypos = np.meshgrid(range(M.shape[0]), range(M.shape[1]))
        xpos = xpos.T.flatten() + 0.5
        ypos = ypos.T.flatten() + 0.5
        zpos = np.zeros(n)
        dx = dy = (1 - options['bars_spacing']) * np.ones(n)

        bar_M = _get_matrix_components(bar_style, M, 'bar_style')

        if isinstance(limits, list) and \
                len(limits) == 2:
            z_min = limits[0]
            z_max = limits[1]
        else:
            if i == 0:
                z_min = min(bar_M)
                z_max = max(bar_M)
            else:
                z_min = min(min(bar_M), z_min)
                z_max = max(max(bar_M), z_max)

            if z_min == z_max:
                z_min -= 0.1
                z_max += 0.1

        color_M = _get_matrix_components(color_style, M, 'color_style')

        if isinstance(color_limits, list) and \
                len(color_limits) == 2:
            c_min = color_limits[0]
            c_max = color_limits[1]
        else:
            if color_style == 'phase':
                c_min = -pi
                c_max = pi
            else:
                if i == 0:
                    c_min = min(color_M)
                    c_max = max(color_M)
                else:
                    c_min = min(min(color_M), c_min)
                    c_max = max(max(color_M), c_max)

            if c_min == c_max:
                c_min -= 0.1
                c_max += 0.1
        _bars_and_colors.append((bar_M, color_M))
    norm = mpl.colors.Normalize(c_min, c_max)

    if cmap is None:
        # change later
        if color_style == 'phase':
            cmap = _cyclic_cmap()
        else:
            cmap = _sequential_cmap()

    artist_list = list()
    for bar_M, color_M in _bars_and_colors:
        colors = cmap(norm(color_M))

        colors[:, 3] = options['bars_alpha']

        if options['threshold'] is not None:
            colors[:, 3] *= 1 * (bar_M >= options['threshold'])

            idx, = np.where(bar_M < options['threshold'])
            bar_M[idx] = 0

        fig, ax = _is_fig_and_ax(fig, ax, projection='3d')

        artist = ax.bar3d(xpos, ypos, zpos, dx, dy, bar_M, color=colors,
                          edgecolors=options['bars_edgecolor'],
                          linewidths=options['bars_lw'],
                          shade=options['shade'])
        artist_list.append([artist])

    # remove vertical lines on xz and yz plane
    ax.yaxis._axinfo["grid"]['linewidth'] = 0
    ax.xaxis._axinfo["grid"]['linewidth'] = 0

    # x axis
    _update_xaxis(options['bars_spacing'], M, ax, x_basis)

    # y axis
    _update_yaxis(options['bars_spacing'], M, ax, y_basis)

    # z axis
    _update_zaxis(ax, z_min, z_max, options['zticks'])

    # stick to xz and yz plane
    _stick_to_planes(options['stick'],
                     options['azim'], ax, M,
                     options['bars_spacing'])
    ax.view_init(azim=options['azim'], elev=options['elev'])

    # color axis
    if colorbar:
        cax, kw = mpl.colorbar.make_axes(ax, shrink=.75,
                                         pad=options['cbar_pad'])
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

        if color_style == 'real':
            cb.set_label('real')
        elif color_style == 'img':
            cb.set_label('imaginary')
        elif color_style == 'abs':
            cb.set_label('absolute')
        else:
            cb.set_label('arg')
            if color_limits is None:
                cb.set_ticks([-pi, -pi / 2, 0, pi / 2, pi])
                cb.set_ticklabels(
                    (r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'))

    # removing margins
    _remove_margins(ax.xaxis)
    _remove_margins(ax.yaxis)
    _remove_margins(ax.zaxis)

    ani = animation.ArtistAnimation(fig, artist_list, interval=50,
                                    blit=True, repeat_delay=1000)
    plt.close()

    ani, html_video = make_html_video(ani, save_options)
    print(z_min, z_max)
    return fig, ani, html_video


def anim_fock_distribution(rhos, fock_numbers=None, color="green",
                           unit_y_range=True, *, fig=None, ax=None,
                           save_options=None):
    """
    Plot the Fock distribution for a density matrix (or ket) that describes
    an oscillator mode.

    Parameters
    ----------
    rhos : list of `qutip.Qobj`
        The density matrices (or kets) to visualize.

    fock_numbers : list of strings, optional
        list of x ticklabels to represent fock numbers

    color : color or list of colors, default="green"
        The colors of the bar faces.

    unit_y_range : bool, default=True
        Set y-axis limits [0, 1] or not

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The axes context in which the plot will be drawn.

    save_options : dict, optional
        A dictionary containing options to save the animation.

        'name' : str, default='animation'
            The output filename, e.g., :file:`mymovie.mp4`.

        'writer' : `MovieWriter` or str, optional
            A `MovieWriter` instance to use or a key that identifies a
            class to use, such as 'ffmpeg'.

        'codec' : str, optional
            The video codec to use.  Not all codecs are supported by a given
            `MovieWriter`.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.
    """

    fig, ax = _is_fig_and_ax(fig, ax)

    artist_list = list()
    for rho in rhos:
        if isket(rho):
            rho = ket2dm(rho)

        N = rho.shape[0]

        artist = ax.bar(np.arange(N), np.real(rho.diag()),
                        color=color, alpha=0.6, width=0.8).patches
        artist_list.append(artist)

    if fock_numbers:
        _set_ticklabels(ax, fock_numbers, np.arange(N), 'x', fontsize=12)

    if unit_y_range:
        ax.set_ylim(0, 1)
    ax.set_xlim(-.5, N)
    ax.set_xlabel('Fock number', fontsize=12)
    ax.set_ylabel('Occupation probability', fontsize=12)

    ani = animation.ArtistAnimation(fig, artist_list, interval=50,
                                    blit=True, repeat_delay=1000)
    plt.close()

    ani, html_video = make_html_video(ani, save_options)

    return fig, ani, html_video


def anim_wigner(rhos, xvec=None, yvec=None, method='clenshaw',
                projection='2d', *, cmap=None, colorbar=False,
                fig=None, ax=None, save_options=None):
    """
    Plot the time evolution of the Wigner function for density matrices
    (or kets) hat describes an oscillator mode.

    Parameters
    ----------
    rhos : list of `qutip.Qobj`
        List of density matrices (or kets) to visualize.

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

    save_options : dict, optional
        A dictionary containing options to save the animation.

        'name' : str, default='animation'
            The output filename, e.g., :file:`mymovie.mp4`.

        'writer' : `MovieWriter` or str, optional
            A `MovieWriter` instance to use or a key that identifies a
            class to use, such as 'ffmpeg'.

        'codec' : str, optional
            The video codec to use.  Not all codecs are supported by a given
            `MovieWriter`.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.
    """

    wlim = 0
    Ws = list()
    artist_list = list()

    if projection not in ('2d', '3d'):
        raise ValueError('Unexpected value of projection keyword argument')
    fig, ax = _is_fig_and_ax(fig, ax, projection)

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
            cf = ax.contourf(xvec, yvec, W, 100, norm=norm,
                             cmap=cmap).collections
        else:
            X, Y = np.meshgrid(xvec, yvec)
            cf = [ax.plot_surface(X, Y, W, rstride=5, cstride=5, linewidth=0.5,
                                  norm=norm, cmap=cmap)]
        artist_list.append(cf)

    ax.set_xlabel(r'$\rm{Re}(\alpha)$', fontsize=12)
    ax.set_ylabel(r'$\rm{Im}(\alpha)$', fontsize=12)

    if colorbar:
        if projection == '2d':
            shrink = 1
        else:
            shrink = .75
        cax, kw = mpl.colorbar.make_axes(ax, shrink=shrink, pad=.1)
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

    ax.set_title("Wigner function", fontsize=12)

    ani = animation.ArtistAnimation(fig, artist_list, interval=50,
                                    blit=True, repeat_delay=1000)
    plt.close()

    ani, html_video = make_html_video(ani, save_options)

    return fig, ani, html_video


def anim_spin_distribution(Ps, THETA, PHI, projection='2d', *,
                           cmap=None, colorbar=False, fig=None,
                           ax=None, save_options=None):
    """
    Plots a spin distribution (given as meshgrid data).

    Parameters
    ----------
    Ps : matrices
        List of distribution values as a meshgrid matrix.

    THETA : matrix
        Meshgrid matrix for the theta coordinate. Its range is between 0 and pi

    PHI : matrix
        Meshgrid matrix for the phi coordinate. Its range is between 0 and 2*pi

    projection: string {'2d', '3d'}, default='2d'
        Specify whether the spin distribution function is to be plotted as a 2D
        projection where the surface of the unit sphere is mapped on
        the unit disk ('2d') or surface plot ('3d').

    cmap : a matplotlib cmap instance, optional
        The colormap.

    colorbar : bool, default=False
        Whether (True) or not (False) a colorbar should be attached to the
        Wigner function graph.

    fig : a matplotlib figure instance, optional
        The figure canvas on which the plot will be drawn.

    ax : a matplotlib axis instance, optional
        The axis context in which the plot will be drawn.

    save_options : dict, optional
        A dictionary containing options to save the animation.

        'name' : str, default='animation'
            The output filename, e.g., :file:`mymovie.mp4`.

        'writer' : `MovieWriter` or str, optional
            A `MovieWriter` instance to use or a key that identifies a
            class to use, such as 'ffmpeg'.

        'codec' : str, optional
            The video codec to use.  Not all codecs are supported by a given
            `MovieWriter`.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.
    """

    if projection in ('2d', '3d'):
        fig, ax = _is_fig_and_ax(fig, ax, projection)
    else:
        raise ValueError('Unexpected value of projection keyword argument')
    min_P = Ps[0].min()
    max_P = Ps[0].max()
    for P in Ps:
        min_P = min(min_P, P.min())
        max_P = max(max_P, P.max())

    if cmap is None:
        if min_P < -1e12:
            cmap = _diverging_cmap()
            norm = mpl.colors.Normalize(-max_P, max_P)
        else:
            cmap = _sequential_cmap()
            norm = mpl.colors.Normalize(min_P, max_P)

    artist_list = list()
    if projection == '2d':
        Y = (THETA - pi / 2) / (pi / 2)
        X = (pi - PHI) / pi * np.sqrt(cos(THETA - pi / 2))
        for P in Ps:
            artist_list.append([ax.pcolor(X, Y, P.real, cmap=cmap)])
        ax.set_xlabel(r'$\varphi$', fontsize=18)
        ax.set_ylabel(r'$\theta$', fontsize=18)
        ax.axis('equal')
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$'], fontsize=18)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels([r'$\pi$', r'$\pi/2$', r'$0$'], fontsize=18)
    else:
        ax.view_init(azim=-35, elev=35)

        xx = sin(THETA) * cos(PHI)
        yy = sin(THETA) * sin(PHI)
        zz = cos(THETA)
        for P in Ps:
            artist = [ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,
                      facecolors=cmap(norm(P)), linewidth=0)]
            artist_list.append(artist)
    if colorbar:
        cax, kw = mpl.colorbar.make_axes(ax, shrink=.66, pad=.1)
        cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb1.set_label('magnitude')

    ani = animation.ArtistAnimation(fig, artist_list, interval=50,
                                    blit=True, repeat_delay=1000)
    plt.close()

    ani, html_video = make_html_video(ani, save_options)

    return fig, ani, html_video


def anim_qubism(kets, theme='light', how='pairs', grid_iteration=1,
                legend_iteration=0, *, fig=None, ax=None, save_options=None):
    """
    Qubism plot for pure states of many qudits.  Works best for spin chains,
    especially with even number of particles of the same dimension.  Allows to
    see entanglement between first 2k particles and the rest.

    .. note::

        colorblind_safe does not apply because of its unique colormap

    Parameters
    ----------
    kets : list of Qobj
        list of pure states for plotting.

    theme : 'light' or 'dark', default='light'
        Set coloring theme for mapping complex values into colors.
        See: complex_array_to_rgb.

    how : 'pairs', 'pairs_skewed' or 'before_after', default='pairs'
        Type of Qubism plotting.  Options:

        - 'pairs' - typical coordinates,
        - 'pairs_skewed' - for ferromagnetic/antriferromagnetic plots,
        - 'before_after' - related to Schmidt plot (see also: plot_schmidt).

    grid_iteration : int, default=1
        Helper lines to be drawn on plot.
        Show tiles for 2*grid_iteration particles vs all others.

    legend_iteration : int or 'grid_iteration' or 'all', default=0
        Show labels for first ``2*legend_iteration`` particles.  Option
        'grid_iteration' sets the same number of particles as for
        grid_iteration.  Option 'all' makes label for all particles.  Typically
        it should be 0, 1, 2 or perhaps 3.

    fig : a matplotlib figure instance, optional
        The figure canvas on which the plot will be drawn.

    ax : a matplotlib axis instance, optional
        The axis context in which the plot will be drawn.

    save_options : dict, optional
        A dictionary containing options to save the animation.

        'name' : str, default='animation'
            The output filename, e.g., :file:`mymovie.mp4`.

        'writer' : `MovieWriter` or str, optional
            A `MovieWriter` instance to use or a key that identifies a
            class to use, such as 'ffmpeg'.

        'codec' : str, optional
            The video codec to use.  Not all codecs are supported by a given
            `MovieWriter`.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.

    Notes
    -----
    See also [1]_.

    References
    ----------
    .. [1] J. Rodriguez-Laguna, P. Migdal, M. Ibanez Berganza, M. Lewenstein
       and G. Sierra, *Qubism: self-similar visualization of many-body
       wavefunctions*, `New J. Phys. 14 053028
       <https://dx.doi.org/10.1088/1367-2630/14/5/053028>`_, arXiv:1112.3560
       (2012), open access.
    """

    fig, ax = _is_fig_and_ax(fig, ax)

    artist_list = list()
    for ket in kets:
        if not isket(ket):
            raise Exception("Qubism works only for pure states, i.e. kets.")
            # add for dm? (perhaps a separate function, plot_qubism_dm)

        dim_list = ket.dims[0]
        n = len(dim_list)

        # for odd number of particles - pixels are rectangular
        if n % 2 == 1:
            ket = tensor(ket, Qobj([1] * dim_list[-1]))
            dim_list = ket.dims[0]
            n += 1

        ketdata = ket.full()

        if how == 'pairs':
            dim_list_y = dim_list[::2]
            dim_list_x = dim_list[1::2]
        elif how == 'pairs_skewed':
            dim_list_y = dim_list[::2]
            dim_list_x = dim_list[1::2]
            if dim_list_x != dim_list_y:
                raise Exception("For 'pairs_skewed' pairs " +
                                "of dimensions need to be the same.")
        elif how == 'before_after':
            dim_list_y = list(reversed(dim_list[:(n // 2)]))
            dim_list_x = dim_list[(n // 2):]
        else:
            raise Exception("No such 'how'.")

        size_x = np.prod(dim_list_x)
        size_y = np.prod(dim_list_y)

        qub = np.zeros([size_x, size_y], dtype=complex)
        for i in range(ketdata.size):
            qub[_to_qubism_index_pair(i, dim_list, how=how)] = ketdata[i, 0]
        qub = qub.transpose()

        quadrants_x = np.prod(dim_list_x[:grid_iteration])
        quadrants_y = np.prod(dim_list_y[:grid_iteration])

        ticks_x = [size_x // quadrants_x * i for i in range(1, quadrants_x)]
        ticks_y = [size_y // quadrants_y * i for i in range(1, quadrants_y)]

        ax.set_xticks(ticks_x)
        ax.set_xticklabels([""] * (quadrants_x - 1))
        ax.set_yticks(ticks_y)
        ax.set_yticklabels([""] * (quadrants_y - 1))
        theme2color_of_lines = {'light': '#000000',
                                'dark': '#FFFFFF'}
        ax.grid(True, color=theme2color_of_lines[theme])
        artist = [ax.imshow(complex_array_to_rgb(qub, theme=theme),
                  interpolation="none",
                  extent=(0, size_x, 0, size_y))]
        artist_list.append(artist)

    if legend_iteration == 'all':
        label_n = n // 2
    elif legend_iteration == 'grid_iteration':
        label_n = grid_iteration
    else:
        try:
            label_n = int(legend_iteration)
        except:
            raise Exception("No such option for legend_iteration keyword " +
                            "argument. Use 'all', 'grid_iteration' or an " +
                            "integer.")

    if label_n:

        if how == 'before_after':
            dim_list_small = list(reversed(dim_list_y[-label_n:])) \
                + dim_list_x[:label_n]
        else:
            dim_list_small = []
            for j in range(label_n):
                dim_list_small.append(dim_list_y[j])
                dim_list_small.append(dim_list_x[j])

        scale_x = float(size_x) / np.prod(dim_list_x[:label_n])
        shift_x = 0.5 * scale_x
        scale_y = float(size_y) / np.prod(dim_list_y[:label_n])
        shift_y = 0.5 * scale_y

        bbox = ax.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted())
        fontsize = 35 * bbox.width / np.prod(dim_list_x[:label_n]) / label_n
        opts = {'fontsize': fontsize,
                'color': theme2color_of_lines[theme],
                'horizontalalignment': 'center',
                'verticalalignment': 'center'}
        for i in range(np.prod(dim_list_small)):
            x, y = _to_qubism_index_pair(i, dim_list_small, how=how)
            seq = _index_to_sequence(i, dim_list=dim_list_small)
            ax.text(scale_x * x + shift_x,
                    size_y - (scale_y * y + shift_y),
                    _sequence_to_latex(seq),
                    **opts)
    ani = animation.ArtistAnimation(fig, artist_list, interval=50,
                                    blit=True, repeat_delay=1000)
    plt.close()

    ani, html_video = make_html_video(ani, save_options)

    return fig, ani, html_video


def anim_schmidt(kets, theme='light', splitting=None,
                 labels_iteration=(3, 2), *, fig=None, ax=None,
                 save_options=None):
    """
    Plotting scheme related to Schmidt decomposition.
    Converts a state into a matrix (A_ij -> A_i^j),
    where rows are first particles and columns - last.

    See also: plot_qubism with how='before_after' for a similar plot.

    .. note::

        colorblind_safe does not apply because of its unique colormap

    Parameters
    ----------
    kets : list of Qobj
        list of pure state for plotting.

    theme : 'light' or 'dark', default='light'
        Set coloring theme for mapping complex values into colors.
        See: complex_array_to_rgb.

    splitting : int, optional
        Plot for a number of first particles versus the rest.
        If not given, it is (number of particles + 1) // 2.

    labels_iteration : int or pair of ints, default=(3,2)
        Number of particles to be shown as tick labels,
        for first (vertical) and last (horizontal) particles, respectively.

    fig : a matplotlib figure instance, optional
        The figure canvas on which the plot will be drawn.

    ax : a matplotlib axis instance, optional
        The axis context in which the plot will be drawn.

    save_options : dict, optional
        A dictionary containing options to save the animation.

        'name' : str, default='animation'
            The output filename, e.g., :file:`mymovie.mp4`.

        'writer' : `MovieWriter` or str, optional
            A `MovieWriter` instance to use or a key that identifies a
            class to use, such as 'ffmpeg'.

        'codec' : str, optional
            The video codec to use.  Not all codecs are supported by a given
            `MovieWriter`.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.

    """

    fig, ax = _is_fig_and_ax(fig, ax)

    artist_list = list()
    for ket in kets:
        if not isket(ket):
            err = "Schmidt plot works only for pure states, i.e. kets."
            raise Exception(err)

        dim_list = ket.dims[0]

        if splitting is None:
            splitting = (len(dim_list) + 1) // 2

        if isinstance(labels_iteration, int):
            labels_iteration = labels_iteration, labels_iteration

        ketdata = ket.full()

        dim_list_y = dim_list[:splitting]
        dim_list_x = dim_list[splitting:]

        size_x = np.prod(dim_list_x)
        size_y = np.prod(dim_list_y)

        ketdata = ketdata.reshape((size_y, size_x))

        artist = [ax.imshow(complex_array_to_rgb(ketdata, theme=theme),
                            interpolation="none",
                            extent=(0, size_x, 0, size_y))]
        artist_list.append(artist)

    dim_list_small_x = dim_list_x[:labels_iteration[1]]
    dim_list_small_y = dim_list_y[:labels_iteration[0]]

    quadrants_x = np.prod(dim_list_small_x)
    quadrants_y = np.prod(dim_list_small_y)

    ticks_x = [size_x / quadrants_x * (i + 0.5)
               for i in range(quadrants_x)]
    ticks_y = [size_y / quadrants_y * (quadrants_y - i - 0.5)
               for i in range(quadrants_y)]

    labels_x = [_sequence_to_latex(_index_to_sequence(i*size_x // quadrants_x,
                                                      dim_list=dim_list_x))
                for i in range(quadrants_x)]
    labels_y = [_sequence_to_latex(_index_to_sequence(i*size_y // quadrants_y,
                                                      dim_list=dim_list_y))
                for i in range(quadrants_y)]

    ax.set_xticks(ticks_x)
    ax.set_xticklabels(labels_x)
    ax.set_yticks(ticks_y)
    ax.set_yticklabels(labels_y)
    ax.set_xlabel("last particles")
    ax.set_ylabel("first particles")

    ani = animation.ArtistAnimation(fig, artist_list, interval=50,
                                    blit=True, repeat_delay=1000)
    plt.close()

    ani, html_video = make_html_video(ani, save_options)

    return fig, ani, html_video