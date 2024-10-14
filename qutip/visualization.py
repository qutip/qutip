"""
Functions for visualizing results of quantum dynamics simulations,
visualizations of quantum states and processes.
"""

__all__ = ['plot_wigner_sphere', 'hinton', 'sphereplot',
           'matrix_histogram', 'plot_energy_levels', 'plot_fock_distribution',
           'plot_wigner', 'plot_expectation_values',
           'plot_spin_distribution', 'complex_array_to_rgb',
           'plot_qubism', 'plot_schmidt']

import itertools as it
import numpy as np
from numpy import pi, array, sin, cos, angle, log2, sqrt

from packaging.version import parse as parse_version

from . import (
    Qobj, isket, ket2dm, tensor, vector_to_operator, settings
)
from .core.superop_reps import _to_superpauli, isqubitdims
from .wigner import wigner
from .matplotlib_utilities import complex_phase_cmap

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.animation as animation
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


def _cyclic_cmap():
    if settings.colorblind_safe:
        return cm.twilight
    else:
        return complex_phase_cmap()


def _diverging_cmap():
    if settings.colorblind_safe:
        return cm.seismic
    else:
        return cm.RdBu


def _sequential_cmap():
    if settings.colorblind_safe:
        return cm.cividis
    else:
        return cm.jet


def _is_fig_and_ax(fig, ax, projection='2d'):
    if fig is None:
        if ax is None:
            fig = plt.figure()
            if projection == '2d':
                ax = fig.add_subplot(1, 1, 1)
            else:
                ax = _axes3D(fig)
        else:
            fig = ax.get_figure()
    else:
        if ax is None:
            if projection == '2d':
                ax = fig.add_subplot(1, 1, 1)
            else:
                ax = _axes3D(fig)

    return fig, ax


def _set_ticklabels(ax, ticklabels, ticks, axis, fontsize=14):
    if len(ticks) != len(ticklabels):
        raise ValueError(
            f"got {len(ticklabels)} ticklabels but needed {len(ticks)}"
        )
    if axis == 'x':
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=fontsize)
    elif axis == 'y':
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels, fontsize=fontsize)
    else:
        raise ValueError(
            "axis must be either 'x' or 'y'"
        )


def _equal_shape(matrices):
    first_shape = matrices[0].shape

    text = "All inputs should have the same shape."
    if not all(matrix.shape == first_shape for matrix in matrices):
        raise ValueError(text)


def plot_wigner_sphere(wigner, reflections=False, *, cmap=None,
                       colorbar=True, fig=None, ax=None):
    """Plots a coloured Bloch sphere.

    Parameters
    ----------
    wigner : a wigner transformation
        The wigner transformation at `steps` different theta and phi.

    reflections : bool, default: False
        If the reflections of the sphere should be plotted as well.

    cmap : a matplotlib colormap instance, optional
        Color map to use when plotting.

    colorbar : bool, default: True
        Whether (True) or not (False) a colorbar should be attached.

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The ax context in which the plot will be drawn.

    Returns
    -------
    fig, output : tuple
        A tuple of the matplotlib figure and the axes instance or animation
        instance used to produce the figure.

    Notes
    -----
    Special thanks to Russell P Rundle for writing this function.
    """

    fig, ax = _is_fig_and_ax(fig, ax, projection='3d')

    if not isinstance(wigner, list):
        wigners = [wigner]
    else:
        wigners = wigner

    _equal_shape(wigners)

    wigner_max = np.real(np.amax(np.abs(wigners[0])))
    for wigner in wigners:
        wigner_max = max(np.real(np.amax(np.abs(wigner))), wigner_max)

    norm = mpl.colors.Normalize(-wigner_max, wigner_max)

    if cmap is None:
        cmap = _diverging_cmap()

    artist_list = list()
    for wigner in wigners:
        steps = len(wigner)
        theta = np.linspace(0, np.pi, steps)
        phi = np.linspace(0, 2 * np.pi, steps)
        x = np.outer(np.sin(theta), np.cos(phi))
        y = np.outer(np.sin(theta), np.sin(phi))
        z = np.outer(np.cos(theta), np.ones(steps))
        wigner = np.real(wigner)

        artist = list()
        # Plot coloured Bloch sphere:
        artist.append(ax.plot_surface(x, y, z, facecolors=cmap(norm(wigner)),
                                      rcount=steps, ccount=steps, linewidth=0,
                                      zorder=0.5, antialiased=None))

        if reflections:
            side_color = cmap(norm(wigner[0:steps, 0:steps]))

            # Plot bottom reflection:
            artist.append(ax.plot_surface(x[0:steps, 0:steps],
                                          y[0:steps, 0:steps],
                                          -1.5*np.ones((steps, steps)),
                                          facecolors=side_color,
                                          rcount=steps/2, ccount=steps/2,
                                          linewidth=0, zorder=0.5,
                                          antialiased=False))

            # Plot side reflection:
            artist.append(ax.plot_surface(-1.5*np.ones((steps, steps)),
                                          y[0:steps, 0:steps],
                                          z[0:steps, 0:steps],
                                          facecolors=side_color,
                                          rcount=steps/2, ccount=steps/2,
                                          linewidth=0, zorder=0.5,
                                          antialiased=False))

            # Plot back reflection:
            artist.append(ax.plot_surface(x[0:steps, 0:steps],
                                          1.5*np.ones((steps, steps)),
                                          z[0:steps, 0:steps],
                                          facecolors=side_color,
                                          rcount=steps/2, ccount=steps/2,
                                          linewidth=0, zorder=0.5,
                                          antialiased=False))
        artist_list.append(artist)

    if len(wigners) == 1:
        output = ax
    else:
        output = animation.ArtistAnimation(fig, artist_list, interval=50,
                                           blit=True, repeat_delay=1000)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Create colourbar:
    if colorbar:
        cax, kw = mpl.colorbar.make_axes(ax, shrink=0.75, pad=.1)
        mpl.colorbar.ColorbarBase(cax, norm=norm, cmap=cmap)

    return fig, output


# Adopted from the SciPy Cookbook.
def _blob(x, y, w, w_max, area, color_fn, ax=None):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = array([x - hs, x + hs, x + hs, x - hs])
    ycorners = array([y - hs, y - hs, y + hs, y + hs])

    if ax is not None:
        handle = ax
    else:
        handle = plt

    return handle.fill(xcorners, ycorners, color=color_fn(w))


def _cb_labels(left_dims):
    """Creates plot labels for matrix elements in the computational basis.

    Parameters
    ----------
    left_dims : flat list of ints
        Dimensions of the left index of a density operator. E. g.
        [2, 3] for a qubit tensored with a qutrit.

    Returns
    -------
    left_labels, right_labels : lists of strings
        Labels for the left and right indices of a density operator
        (kets and bras, respectively).
    """
    # FIXME: assumes dims, such that we only need left_dims == dims[0].
    basis_labels = list(map(",".join, it.product(*[
        map(str, range(dim))
        for dim in left_dims
    ])))
    return [
        map(fmt.format, basis_labels) for fmt in
        (
            r"$\langle{}|$",
            r"$|{}\rangle$",
        )
    ]


# Adopted from the SciPy Cookbook.
def hinton(rho, x_basis=None, y_basis=None, color_style="scaled",
           label_top=True, *, cmap=None, colorbar=True, fig=None, ax=None):
    """Draws a Hinton diagram to visualize a density matrix or superoperator.

    Parameters
    ----------
    rho : qobj
        Input density matrix or superoperator.

        .. note::

            Hinton plots of superoperators are currently only
            supported for qubits.

    x_basis : list of strings, optional
        list of x ticklabels to represent x basis of the input.

    y_basis : list of strings, optional
        list of y ticklabels to represent y basis of the input.

    color_style : str, {"scaled", "threshold", "phase"}, default: "scaled"

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

    label_top : bool, default: True
        If True, x ticklabels will be placed on top, otherwise
        they will appear below the plot.

    cmap : a matplotlib colormap instance, optional
        Color map to use when plotting.

    colorbar : bool, default: True
        Whether (True) or not (False) a colorbar should be attached.

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The ax context in which the plot will be drawn.

    Returns
    -------
    fig, output : tuple
        A tuple of the matplotlib figure and the axes instance or animation
        instance used to produce the figure.

    Raises
    ------
    ValueError
        Input argument is not a quantum object.

    Examples
    --------
    >>> import qutip

    >>> dm = qutip.rand_dm(4)
    >>> fig, ax = qutip.hinton(dm)
    >>> fig.show()

    >>> qutip.settings.colorblind_safe = True
    >>> fig, ax = qutip.hinton(dm, color_style="threshold")
    >>> fig.show()
    >>> qutip.settings.colorblind_safe = False

    >>> fig, ax = qutip.hinton(dm, color_style="phase")
    >>> fig.show()
    """

    fig, ax = _is_fig_and_ax(fig, ax)

    if not isinstance(rho, list):
        rhos = [rho]
    else:
        rhos = rho

    _equal_shape(rhos)

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

    if len(rhos) == 1:
        output = ax
    else:
        output = animation.ArtistAnimation(fig, artist_list, interval=50,
                                           blit=True, repeat_delay=1000)

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

    if colorbar:
        vmax = np.pi if color_style == "phase" else w_max
        norm = mpl.colors.Normalize(-vmax, vmax)
        cax, kw = mpl.colorbar.make_axes(ax, shrink=0.75, pad=.1)
        mpl.colorbar.ColorbarBase(cax, norm=norm, cmap=cmap)

    return fig, output


def sphereplot(values, theta, phi, *,
               cmap=None, colorbar=True, fig=None, ax=None):
    """Plots a matrix of values on a sphere

    Parameters
    ----------
    values : array
        Data set to be plotted

    theta : float
        Angle with respect to z-axis. Its range is between 0 and pi

    phi : float
        Angle in x-y plane. Its range is between 0 and 2*pi

    cmap : a matplotlib colormap instance, optional
        Color map to use when plotting.

    colorbar : bool, default: True
        Whether (True) or not (False) a colorbar should be attached.

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The axes context in which the plot will be drawn.

    Returns
    -------
    fig, output : tuple
        A tuple of the matplotlib figure and the axes instance or animation
        instance used to produce the figure.
    """

    fig, ax = _is_fig_and_ax(fig, ax, projection='3d')

    if not isinstance(values, list):
        V = [values]
    else:
        V = values

    _equal_shape(V)

    r_and_ph = list()
    min_ph = pi
    max_ph = -pi
    for values in V:
        r = array(abs(values))
        ph = angle(values)
        min_ph = min(min_ph, ph.min())
        max_ph = max(max_ph, ph.max())
        r_and_ph.append((r, ph))

    # normalize color range based on phase angles in list ph
    norm = mpl.colors.Normalize(min_ph, max_ph)

    if cmap is None:
        cmap = _sequential_cmap()

    # plot with facecolors set to cm.jet colormap normalized to nrm
    thetam, phim = np.meshgrid(theta, phi)
    xx = sin(thetam) * cos(phim)
    yy = sin(thetam) * sin(phim)
    zz = cos(thetam)
    artist_list = list()
    for r, ph in r_and_ph:
        artist = [ax.plot_surface(r * xx, r * yy, r * zz, rstride=1, cstride=1,
                                  facecolors=cmap(norm(ph)), linewidth=0,)]
        artist_list.append(artist)

    if len(V) == 1:
        output = ax
    else:
        output = animation.ArtistAnimation(fig, artist_list, interval=50,
                                           blit=True, repeat_delay=1000)

    if colorbar:
        # create new axes on plot for colorbar and shrink it a bit.
        # pad shifts location of bar with repsect to the main plot
        cax, kw = mpl.colorbar.make_axes(ax, shrink=.66, pad=.05)

        # create new colorbar in axes cax with cmap and normalized to nrm like
        # our facecolors
        cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        # add our colorbar label
        cb1.set_label('Angle')

    return fig, output


def _remove_margins(axis):
    """
    removes margins about z = 0 and improves the style
    by monkey patching
    """

    def _get_coord_info_new_mpl38(renderer):
        mins, maxs, centers, deltas, tc, highs = _get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs

    def _get_coord_info_new_mpl39():
        mins, maxs, bounds_proj, highs = _get_coord_info_old()
        centers, deltas = axis._calc_centers_deltas(maxs, mins)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, bounds_proj, highs

    _get_coord_info_old = axis._get_coord_info

    # Select correct version of the function based on matplotlib version
    if parse_version(mpl.__version__) >= parse_version("3.9"):
        axis._get_coord_info = _get_coord_info_new_mpl39
    else:
        axis._get_coord_info = _get_coord_info_new_mpl38


def _stick_to_planes(stick, azim, ax, M, spacing):
    """adjusts xlim and ylim in way that bars will
    stick to xz and yz planes
    """
    if stick is True:
        azim = azim % 360
        if 0 <= azim <= 90:
            ax.set_ylim(1 - .5,)
            ax.set_xlim(1 - .5,)
        elif 90 < azim <= 180:
            ax.set_ylim(1 - .5,)
            ax.set_xlim(0, M.shape[0] + (.5 - spacing))
        elif 180 < azim <= 270:
            ax.set_ylim(0, M.shape[1] + (.5 - spacing))
            ax.set_xlim(0, M.shape[0] + (.5 - spacing))
        elif 270 < azim < 360:
            ax.set_ylim(0, M.shape[1] + (.5 - spacing))
            ax.set_xlim(1 - .5,)


def _update_yaxis(spacing, M, ax, ylabels):
    """
    updates the y-axis
    """
    ytics = [y + (1 - (spacing / 2)) for y in range(M.shape[1])]
    ax.yaxis.set_major_locator(plt.FixedLocator(ytics))
    if ylabels:
        nylabels = len(ylabels)
        if nylabels != len(ytics):
            raise ValueError(f"got {nylabels} ylabels but needed {len(ytics)}")
        ax.set_yticklabels(ylabels)
    else:
        ax.set_yticklabels([str(y + 1) for y in range(M.shape[1])])
        ax.set_yticklabels([str(i) for i in range(M.shape[1])])
    ax.tick_params(axis='y', labelsize=14)
    ax.set_yticks([y + (1 - (spacing / 2)) for y in range(M.shape[1])])


def _update_xaxis(spacing, M, ax, xlabels):
    """
    updates the x-axis
    """
    xtics = [x + (1 - (spacing / 2)) for x in range(M.shape[0])]
    ax.xaxis.set_major_locator(plt.FixedLocator(xtics))
    if xlabels:
        nxlabels = len(xlabels)
        if nxlabels != len(xtics):
            raise ValueError(f"got {nxlabels} xlabels but needed {len(xtics)}")
        ax.set_xticklabels(xlabels)
    else:
        ax.set_xticklabels([str(x + 1) for x in range(M.shape[0])])
        ax.set_xticklabels([str(i) for i in range(M.shape[0])])
    ax.tick_params(axis='x', labelsize=14)
    ax.set_xticks([x + (1 - (spacing / 2)) for x in range(M.shape[0])])


def _update_zaxis(ax, z_min, z_max, zticks):
    """
    updates the z-axis
    """
    ax.zaxis.set_major_locator(plt.IndexLocator(1, 0.5))
    if isinstance(zticks, list):
        ax.set_zticks(zticks)
    ax.set_zlim3d([min(z_min, 0), z_max])


def _get_matrix_components(option, M, argument):
    if option == 'real':
        return np.real(M.flatten())
    elif option == 'img':
        return np.imag(M.flatten())
    elif option == 'abs':
        return np.abs(M.flatten())
    elif option == 'phase':
        return angle(M.flatten())
    else:
        raise ValueError("got an unexpected argument, "
                         f"{option} for {argument}")


def sph2cart(r, theta, phi):
    """spherical to cartesian transformation."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def sphview(ax):
    """
    returns the camera position for 3D axes in spherical coordinates."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    # Compute  based on the plots xyz limits.
    r = 0.5 * np.sqrt(
        (xlim[1] - xlim[0]) ** 2 +
        (ylim[1] - ylim[0]) ** 2 +
        (zlim[1] - zlim[0]) ** 2
    )
    theta, phi = np.radians((90 - ax.elev, ax.azim))
    return r, theta, phi


def get_camera_position(ax):
    """
    returns the camera position for 3D axes in cartesian coordinates
    as a 3d numpy array.
    """
    r, theta, phi = sphview(ax)
    return np.array(sph2cart(r, theta, phi), ndmin=3).T


def matrix_histogram(
    M,
    x_basis=None,
    y_basis=None,
    limits=None,
    bar_style="real",
    color_limits=None,
    color_style="real",
    options=None,
    *,
    cmap=None,
    colorbar=True,
    fig=None,
    ax=None,
):
    """
    Draw a histogram for the matrix M, with the given x and y labels and title.

    Parameters
    ----------
    M : Matrix of Qobj
        The matrix to visualize

    x_basis : list of strings, optional
        list of x ticklabels

    y_basis : list of strings, optional
        list of y ticklabels

    limits : list/array with two float numbers, optional
        The z-axis limits [min, max]

    bar_style : str, {"real", "img", "abs", "phase"}, default: "real"

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

    color_style : str, {"real", "img", "abs", "phase"}, default: "real"
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

    colorbar : bool, default: True
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

        'bars_spacing' : float, default: 0.1
            spacing between bars.

        'bars_alpha' : float, default: 1.
            transparency of bars, should be in range 0 - 1

        'bars_lw' : float, default: 0.5
            linewidth of bars' edges.

        'bars_edgecolor' : color, default: 'k'
            The colors of the bars' edges.
            Examples: 'k', (0.1, 0.2, 0.5) or '#0f0f0f80'.

        'shade' : bool, default: True
            Whether to shade the dark sides of the bars (True) or not (False).
            The shading is relative to plot's source of light.

        'azim' : float, default: -35
            The azimuthal viewing angle.

        'elev' : float, default: 35
            The elevation viewing angle.

        'stick' : bool, default: False
            Changes xlim and ylim in such a way that bars next to
            XZ and YZ planes will stick to those planes.
            This option has no effect if ``ax`` is passed as a parameter.

        'cbar_pad' : float, default: 0.04
            The fraction of the original axes between the colorbar
            and the new image axes.
            (i.e. the padding between the 3D figure and the colorbar).

        'cbar_to_z' : bool, default: False
            Whether to set the color of maximum and minimum z-values to the
            maximum and minimum colors in the colorbar (True) or not (False).

        'threshold': float, optional
            Threshold for when bars of smaller height should be transparent. If
            not set, all bars are colored according to the color map.

    Returns
    -------
    fig, output : tuple
        A tuple of the matplotlib figure and the axes instance or animation
        instance used to produce the figure.

    Raises
    ------
    ValueError
        Input argument is not valid.

    """

    default_opts = {
        "zticks": None,
        "bars_spacing": 0.3,
        "bars_alpha": 1.0,
        "bars_lw": 0.7,
        "bars_edgecolor": "k",
        "shade": True,
        "azim": -60,
        "elev": 30,
        "stick": False,
        "cbar_pad": 0.04,
        "cbar_to_z": False,
        "threshold": None,
    }

    # update default_opts from input options
    if options is None:
        options = dict()

    if isinstance(options, dict):
        # check if keys in options dict are valid
        if set(options) - set(default_opts):
            raise ValueError(
                "invalid key(s) found in options: "
                f"{', '.join(set(options) - set(default_opts))}"
            )
        else:
            # updating default options
            default_opts.update(options)
            options = default_opts
    else:
        raise ValueError("options must be a dictionary")

    fig, ax = _is_fig_and_ax(fig, ax, projection="3d")

    if not isinstance(M, list):
        Ms = [M]
    else:
        Ms = M

    _equal_shape(Ms)

    for i, M in enumerate(Ms):
        if isinstance(M, Qobj):
            if x_basis is None:
                x_basis = list(_cb_labels([M.shape[0]])[0])
            if y_basis is None:
                y_basis = list(_cb_labels([M.shape[1]])[1])
            # extract matrix data from Qobj
            M = M.full()

        bar_M = _get_matrix_components(bar_style, M, "bar_style")

        if isinstance(limits, list) and len(limits) == 2:
            z_min = limits[0]
            z_max = limits[1]
        else:
            z_min = min(bar_M) if i == 0 else min(min(bar_M), z_min)
            z_max = max(bar_M) if i == 0 else max(max(bar_M), z_max)

            if z_min == z_max:
                z_min -= 0.1
                z_max += 0.1

        color_M = _get_matrix_components(color_style, M, "color_style")

        if isinstance(color_limits, list) and len(color_limits) == 2:
            c_min = color_limits[0]
            c_max = color_limits[1]
        else:
            if color_style == "phase":
                c_min = -pi
                c_max = pi
            else:
                c_min = min(color_M) if i == 0 else min(min(color_M), c_min)
                c_max = max(color_M) if i == 0 else max(max(color_M), c_max)

            if c_min == c_max:
                c_min -= 0.1
                c_max += 0.1

    norm = mpl.colors.Normalize(c_min, c_max)

    if cmap is None:
        # change later
        if color_style == "phase":
            cmap = _cyclic_cmap()
        else:
            cmap = _sequential_cmap()

    artist_list = list()

    ax.view_init(azim=options['azim'], elev=options['elev'])

    camera = get_camera_position(ax)
    for M in Ms:

        if isinstance(M, Qobj):
            M = M.full()

        bar_M = _get_matrix_components(bar_style, M, "bar_style")
        color_M = _get_matrix_components(color_style, M, "color_style")

        n = np.size(M)
        xpos, ypos = np.meshgrid(range(M.shape[0]), range(M.shape[1]))
        xpos = xpos.T.flatten() + 0.5
        ypos = ypos.T.flatten() + 0.5
        zpos = np.zeros(n)
        dx = dy = (1 - options["bars_spacing"]) * np.ones(n)
        colors = cmap(norm(color_M))

        colors[:, 3] = options["bars_alpha"]

        if options["threshold"] is not None:
            colors[:, 3] *= 1 * (bar_M >= options["threshold"])

            (idx,) = np.where(bar_M < options["threshold"])
            bar_M[idx] = 0

        temp_xpos = xpos.reshape(M.shape)
        temp_ypos = ypos.reshape(M.shape)
        temp_zpos = zpos.reshape(M.shape)

        # calculating z_order for each bar based on its position
        # The sorting issue was fixed by making minor change to
        # https://stackoverflow.com/questions/18602660/matplotlib-bar3d-clipping-problems
        z_order = (
            np.multiply(
                [
                    temp_xpos, temp_ypos, temp_zpos], camera
                    ).sum(0).flatten()
        )

        for i, uxpos in enumerate(xpos):
            artist = ax.bar3d(
                uxpos,
                ypos[i],
                zpos[i],
                dx[i],
                dy[i],
                bar_M[i],
                color=colors[i],
                edgecolors=options["bars_edgecolor"],
                linewidths=options["bars_lw"],
                shade=options["shade"],
            )
            # Setting the z-order for rendering
            artist._sort_zpos = z_order[i]
            artist_list.append([artist])

    if len(Ms) == 1:
        output = ax
    else:
        output = animation.ArtistAnimation(
            fig, artist_list, interval=50, blit=True, repeat_delay=1000
        )

    # remove vertical lines on xz and yz plane
    ax.yaxis._axinfo["grid"]["linewidth"] = 0
    ax.xaxis._axinfo["grid"]["linewidth"] = 0

    # x axis
    _update_xaxis(options["bars_spacing"], M, ax, x_basis)

    # y axis
    _update_yaxis(options["bars_spacing"], M, ax, y_basis)

    # z axis
    _update_zaxis(ax, z_min, z_max, options["zticks"])

    # stick to xz and yz plane
    _stick_to_planes(options["stick"], options["azim"], ax, M, options["bars_spacing"])

    # removing margins
    _remove_margins(ax.xaxis)
    _remove_margins(ax.yaxis)
    _remove_margins(ax.zaxis)

    # color axis
    if colorbar:
        cax, kw = mpl.colorbar.make_axes(
            ax, shrink=0.75, pad=options["cbar_pad"])
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

        if color_style == "real":
            cb.set_label("real")
        elif color_style == "img":
            cb.set_label("imaginary")
        elif color_style == "abs":
            cb.set_label("absolute")
        else:
            cb.set_label("arg")
            if color_limits is None:
                cb.set_ticks([-pi, -pi / 2, 0, pi / 2, pi])
                cb.set_ticklabels(
                    (r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$")
                )

    return fig, output


def plot_energy_levels(H_list, h_labels=None, energy_levels=None, N=0, *,
                       fig=None, ax=None):
    """
    Plot the energy level diagrams for a list of Hamiltonians. Include
    up to N energy levels. For each element in H_list, the energy
    levels diagram for the cummulative Hamiltonian sum(H_list[0:n]) is plotted,
    where n is the index of an element in H_list.

    Parameters
    ----------

        H_list : List of Qobj
            A list of Hamiltonians.

        h_lables : List of string, optional
            A list of xticklabels for each Hamiltonian

        energy_levels : List of string, optional
            A list of  yticklabels to the left of energy levels of the initial
            Hamiltonian.

        N : int, default: 0
            The number of energy levels to plot

        fig : a matplotlib Figure instance, optional
            The Figure canvas in which the plot will be drawn.

        ax : a matplotlib axes instance, optional
            The axes context in which the plot will be drawn.

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

    if not isinstance(H_list, list):
        raise ValueError("H_list must be a list of Qobj instances")

    fig, ax = _is_fig_and_ax(fig, ax)

    H = H_list[0]
    N = H.shape[0] if N == 0 else min(H.shape[0], N)
    xticks = []
    yticks = []
    x = 0
    evals0 = H.eigenenergies(eigvals=N)
    for e_idx, e in enumerate(evals0[:N]):
        ax.plot([x, x + 2], np.array([1, 1]) * e, 'b', linewidth=2)
        yticks.append(e)
    xticks.append(x + 1)
    x += 2

    for H1 in H_list[1:]:

        H = H + H1
        evals1 = H.eigenenergies()

        for e_idx, e in enumerate(evals1[:N]):
            ax.plot([x, x + 1], np.array([evals0[e_idx], e]), 'k:')
        x += 1

        for e_idx, e in enumerate(evals1[:N]):
            ax.plot([x, x + 2], np.array([1, 1]) * e, 'b', linewidth=2)
        xticks.append(x + 1)
        x += 2

        evals0 = evals1

    ax.set_frame_on(False)

    if energy_levels:
        yticks = np.unique(np.around(yticks, 1))
        _set_ticklabels(ax, energy_levels, yticks, 'y')
    else:
        # show eigenenergies
        yticks = np.unique(np.around(yticks, 1))
        ax.set_yticks(yticks)

    if h_labels:
        ax.get_xaxis().tick_bottom()
        _set_ticklabels(ax, h_labels, xticks, 'x')
    else:
        # hide xtick
        ax.tick_params(axis='x', which='both',
                       bottom=False, labelbottom=False)

    return fig, ax


def plot_fock_distribution(rho, fock_numbers=None, color="green",
                           unit_y_range=True, *, fig=None, ax=None):
    """
    Plot the Fock distribution for a density matrix (or ket) that describes
    an oscillator mode.

    Parameters
    ----------
    rho : :obj:`.Qobj`
        The density matrix (or ket) of the state to visualize.

    fock_numbers : list of strings, optional
        list of x ticklabels to represent fock numbers

    color : color or list of colors, default: "green"
        The colors of the bar faces.

    unit_y_range : bool, default: True
        Set y-axis limits [0, 1] or not

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The axes context in which the plot will be drawn.

    Returns
    -------
    fig, output : tuple
        A tuple of the matplotlib figure and the axes instance or animation
        instance used to produce the figure.
    """

    fig, ax = _is_fig_and_ax(fig, ax)

    if not isinstance(rho, list):
        rhos = [rho]
    else:
        rhos = rho

    _equal_shape(rhos)

    artist_list = list()
    for rho in rhos:
        if isket(rho):
            rho = ket2dm(rho)

        N = rho.shape[0]

        artist = ax.bar(np.arange(N), np.real(rho.diag()),
                        color=color, alpha=0.6, width=0.8).patches
        artist_list.append(artist)

    if len(rhos) == 1:
        output = ax
    else:
        output = animation.ArtistAnimation(fig, artist_list, interval=50,
                                           blit=True, repeat_delay=1000)

    if fock_numbers:
        _set_ticklabels(ax, fock_numbers, np.arange(N), 'x', fontsize=12)

    if unit_y_range:
        ax.set_ylim(0, 1)

    ax.set_xlim(-.5, N)
    ax.set_xlabel('Fock number', fontsize=12)
    ax.set_ylabel('Occupation probability', fontsize=12)

    return fig, output


def plot_wigner(rho, xvec=None, yvec=None, method='clenshaw', projection='2d',
                g=sqrt(2), sparse=False, parfor=False, *,
                cmap=None, colorbar=False, fig=None, ax=None):
    """
    Plot the the Wigner function for a density matrix (or ket) that describes
    an oscillator mode.

    Parameters
    ----------
    rho : :obj:`.Qobj`
        The density matrix (or ket) of the state to visualize.

    xvec : array_like, optional
        x-coordinates at which to calculate the Wigner function.

    yvec : array_like, optional
        y-coordinates at which to calculate the Wigner function.  Does not
        apply to the 'fft' method.

    method : str {'clenshaw', 'iterative', 'laguerre', 'fft'}, default: 'clenshaw'
        The method used for calculating the wigner function. See the
        documentation for qutip.wigner for details.

    projection: str {'2d', '3d'}, default: '2d'
        Specify whether the Wigner function is to be plotted as a
        contour graph ('2d') or surface plot ('3d').

    g : float
        Scaling factor for `a = 0.5 * g * (x + iy)`, default `g = sqrt(2)`.
        See the documentation for qutip.wigner for details.

    sparse : bool {False, True}
        Flag for sparse format.
        See the documentation for qutip.wigner for details.

    parfor : bool {False, True}
        Flag for parallel calculation.
        See the documentation for qutip.wigner for details.

    cmap : a matplotlib cmap instance, optional
        The colormap.

    colorbar : bool, default: False
        Whether (True) or not (False) a colorbar should be attached to the
        Wigner function graph.

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The axes context in which the plot will be drawn.

    Returns
    -------
    fig, output : tuple
        A tuple of the matplotlib figure and the axes instance or animation
        instance used to produce the figure.
    """

    if projection not in ('2d', '3d'):
        raise ValueError('Unexpected value of projection keyword argument')

    fig, ax = _is_fig_and_ax(fig, ax, projection)

    if not isinstance(rho, list):
        rhos = [rho]
    else:
        rhos = rho

    _equal_shape(rhos)

    wlim = 0
    Ws = list()
    xvec = np.linspace(-7.5, 7.5, 200) if xvec is None else xvec
    yvec = np.linspace(-7.5, 7.5, 200) if yvec is None else yvec
    for rho in rhos:
        if isket(rho):
            rho = ket2dm(rho)

        W0 = wigner(
            rho, xvec, yvec, method=method,
            g=g, sparse=sparse, parfor=parfor
        )

        W, yvec = W0 if isinstance(W0, tuple) else (W0, yvec)
        Ws.append(W)

        wlim = max(abs(W).max(), wlim)

    norm = mpl.colors.Normalize(-wlim, wlim)

    if cmap is None:
        cmap = _diverging_cmap()

    artist_list = list()
    for W in Ws:
        if projection == '2d':
            if parse_version(mpl.__version__) >= parse_version('3.8'):
                cf = [ax.contourf(xvec, yvec, W, 100, norm=norm, cmap=cmap)]
            else:
                cf = ax.contourf(xvec, yvec, W, 100, norm=norm,
                                 cmap=cmap).collections
        else:
            X, Y = np.meshgrid(xvec, yvec)
            cf = [ax.plot_surface(X, Y, W, rstride=5, cstride=5, linewidth=0.5,
                                  norm=norm, cmap=cmap)]
        artist_list.append(cf)

    if len(rhos) == 1:
        output = ax
    else:
        output = animation.ArtistAnimation(fig, artist_list, interval=50,
                                           blit=True, repeat_delay=1000)

    ax.set_xlabel(r'$\rm{Re}(\alpha)$', fontsize=12)
    ax.set_ylabel(r'$\rm{Im}(\alpha)$', fontsize=12)

    if colorbar:
        if projection == '2d':
            shrink = 1
        else:
            shrink = .75
        cax, kw = mpl.colorbar.make_axes(ax, shrink=shrink, pad=.1)
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

    return fig, output


def plot_expectation_values(results, ylabels=None, *,
                            fig=None, axes=None):
    """
    Visualize the results (expectation values) for an evolution solver.
    `results` is assumed to be an instance of Result, or a list of Result
    instances.

    Parameters
    ----------
    results : (list of) :class:`.Result`
        List of results objects returned by any of the QuTiP evolution solvers.

    ylabels : list of strings, optional
        The y-axis labels. List should be of the same length as `results`.

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    axes : (list of)  axes instances, optional
        The axes context in which the plot will be drawn.

    Returns
    -------
    fig, axes : tuple
        A tuple of the matplotlib figure and array of axes instances
        used to produce the figure.
    """
    if not isinstance(results, list):
        results = [results]

    n_e_ops = max([len(result.expect) for result in results])

    if axes is None:
        if fig is None:
            fig = plt.figure()
        axes = np.array([fig.add_subplot(n_e_ops, 1, i+1)
                         for i in range(n_e_ops)])

    # create np.ndarray if axes is one axes object or list
    if not isinstance(axes, np.ndarray):
        if not isinstance(axes, list):
            axes = [axes]
        axes = np.array(axes)

    for _, result in enumerate(results):
        for e_idx, e in enumerate(result.expect):
            axes[e_idx].plot(result.times, e,
                             label="%s [%d]" % (result.solver, e_idx))

    axes[n_e_ops - 1].set_xlabel("time", fontsize=12)
    for n in range(n_e_ops):
        if ylabels:
            axes[n].set_ylabel(ylabels[n], fontsize=12)

    return fig, axes


def plot_spin_distribution(P, THETA, PHI, projection='2d', *,
                           cmap=None, colorbar=False, fig=None, ax=None):
    """
    Plots a spin distribution (given as meshgrid data).

    Parameters
    ----------
    P : matrix
        Distribution values as a meshgrid matrix.

    THETA : matrix
        Meshgrid matrix for the theta coordinate. Its range is between 0 and pi

    PHI : matrix
        Meshgrid matrix for the phi coordinate. Its range is between 0 and 2*pi

    projection: str {'2d', '3d'}, default: '2d'
        Specify whether the spin distribution function is to be plotted as a 2D
        projection where the surface of the unit sphere is mapped on
        the unit disk ('2d') or surface plot ('3d').

    cmap : a matplotlib cmap instance, optional
        The colormap.

    colorbar : bool, default: False
        Whether (True) or not (False) a colorbar should be attached to the
        Wigner function graph.

    fig : a matplotlib figure instance, optional
        The figure canvas on which the plot will be drawn.

    ax : a matplotlib axis instance, optional
        The axis context in which the plot will be drawn.

    Returns
    -------
    fig, output : tuple
        A tuple of the matplotlib figure and the axes instance or animation
        instance used to produce the figure.
    """

    if projection in ('2d', '3d'):
        fig, ax = _is_fig_and_ax(fig, ax, projection)
    else:
        raise ValueError('Unexpected value of projection keyword argument')

    if not isinstance(P, list):
        Ps = [P]
    else:
        Ps = P

    _equal_shape(Ps)

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
        xx = sin(THETA) * cos(PHI)
        yy = sin(THETA) * sin(PHI)
        zz = cos(THETA)
        for P in Ps:
            artist = [ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,
                      facecolors=cmap(norm(P)), linewidth=0)]
            artist_list.append(artist)
        ax.view_init(azim=-35, elev=35)

    if len(Ps) == 1:
        output = ax
    else:
        output = animation.ArtistAnimation(fig, artist_list, interval=50,
                                           blit=True, repeat_delay=1000)

    if colorbar:
        cax, _ = mpl.colorbar.make_axes(ax, shrink=.66, pad=.1)
        cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb1.set_label('magnitude')

    return fig, output


#
# Qubism and other qubistic visualizations
#
def complex_array_to_rgb(X, theme='light', rmax=None):
    """
    Makes an array of complex number and converts it to an array of [r, g, b],
    where phase gives hue and saturation/value are given by the absolute value.
    Especially for use with imshow for complex plots.

    For more info on coloring, see:
        Emilia Petrisor,
        Visualizing complex-valued functions with Matplotlib and Mayavi
        https://nbviewer.ipython.org/github/empet/Math/blob/master/DomainColoring.ipynb

    Parameters
    ----------
    X : array
        Array (of any dimension) of complex numbers.

    theme : str {'light', 'dark'}, default: 'light'
        Set coloring theme for mapping complex values into colors.

    rmax : float, optional
        Maximal abs value for color normalization.
        If None (default), uses np.abs(X).max().

    Returns
    -------
    Y : array
        Array of colors (of shape X.shape + (3,)).

    """

    absmax = rmax or np.abs(X).max()
    if absmax == 0.:
        absmax = 1.
    Y = np.zeros(X.shape + (3,), dtype='float')
    Y[..., 0] = np.angle(X) / (2 * pi) % 1
    if theme == 'light':
        Y[..., 1] = np.clip(np.abs(X) / absmax, 0, 1)
        Y[..., 2] = 1
    elif theme == 'dark':
        Y[..., 1] = 1
        Y[..., 2] = np.clip(np.abs(X) / absmax, 0, 1)
    Y = mpl.colors.hsv_to_rgb(Y)
    return Y


def _index_to_sequence(i, dim_list):
    """
    For a matrix entry with index i it returns state it corresponds to.
    In particular, for dim_list=[2]*n it returns i written as a binary number.

    Parameters
    ----------
    i : int
        Index in a matrix.

    dim_list : list of int
        List of dimensions of consecutive particles.

    Returns
    -------
    seq : list
        List of coordinates for each particle.

    """
    res = []
    j = i
    for d in reversed(dim_list):
        j, s = divmod(j, d)
        res.append(s)
    return list(reversed(res))


def _sequence_to_index(seq, dim_list):
    """
    Inverse of _index_to_sequence.

    Parameters
    ----------
    seq : list of ints
        List of coordinates for each particle.

    dim_list : list of int
        List of dimensions of consecutive particles.

    Returns
    -------
    i : list
        Index in a matrix.

    """
    i = 0
    for s, d in zip(seq, dim_list):
        i *= d
        i += s

    return i


def _to_qubism_index_pair(i, dim_list, how='pairs'):
    """
    For a matrix entry with index i
    it returns x, y coordinates in qubism mapping.

    Parameters
    ----------
    i : int
        Index in a matrix.

    dim_list : list of int
        List of dimensions of consecutive particles.

    how : 'pairs' ('default'), 'pairs_skewed' or 'before_after'
        Type of qubistic plot.

    Returns
    -------
    x, y : tuple of ints
        List of coordinates for each particle.

    """
    seq = _index_to_sequence(i, dim_list)

    if how == 'pairs':
        y = _sequence_to_index(seq[::2], dim_list[::2])
        x = _sequence_to_index(seq[1::2], dim_list[1::2])
    elif how == 'pairs_skewed':
        dim_list2 = dim_list[::2]
        y = _sequence_to_index(seq[::2], dim_list2)
        seq2 = [(b - a) % d for a, b, d in zip(seq[::2], seq[1::2], dim_list2)]
        x = _sequence_to_index(seq2, dim_list2)
    elif how == 'before_after':
        # https://en.wikipedia.org/wiki/File:Ising-tartan.png
        n = len(dim_list)
        y = _sequence_to_index(reversed(seq[:(n // 2)]),
                               reversed(dim_list[:(n // 2)]))
        x = _sequence_to_index(seq[(n // 2):], dim_list[(n // 2):])
    else:
        raise Exception("No such 'how'.")

    return x, y


def _sequence_to_latex(seq, style='ket'):
    """
    For a sequence of particle states generate LaTeX code.

    Parameters
    ----------
    seq : list of ints
        List of coordinates for each particle.

    style : 'ket' (default), 'bra' or 'bare'
        Style of LaTeX (i.e. |01> or <01| or 01, respectively).

    Returns
    -------
    latex : str
        LaTeX output.

    """
    if style == 'ket':
        latex = "$\\left|{0}\\right\\rangle$"
    elif style == 'bra':
        latex = "$\\left\\langle{0}\\right|$"
    elif style == 'bare':
        latex = "${0}$"
    else:
        raise Exception("No such style.")
    return latex.format("".join(map(str, seq)))


def plot_qubism(ket, theme='light', how='pairs', grid_iteration=1,
                legend_iteration=0, *, fig=None, ax=None):
    """
    Qubism plot for pure states of many qudits.  Works best for spin chains,
    especially with even number of particles of the same dimension.  Allows to
    see entanglement between first 2k particles and the rest.

    .. note::

        colorblind_safe does not apply because of its unique colormap

    Parameters
    ----------
    ket : Qobj
        Pure state for plotting.

    theme : str {'light', 'dark'}, default: 'light'
        Set coloring theme for mapping complex values into colors.
        See: complex_array_to_rgb.

    how : str {'pairs', 'pairs_skewed' or 'before_after'}, default: 'pairs'
        Type of Qubism plotting.  Options:

        - 'pairs' - typical coordinates,
        - 'pairs_skewed' - for ferromagnetic/antriferromagnetic plots,
        - 'before_after' - related to Schmidt plot (see also: plot_schmidt).

    grid_iteration : int, default: 1
        Helper lines to be drawn on plot.
        Show tiles for 2*grid_iteration particles vs all others.

    legend_iteration : int or 'grid_iteration' or 'all', default: 0
        Show labels for first ``2*legend_iteration`` particles.  Option
        'grid_iteration' sets the same number of particles as for
        grid_iteration.  Option 'all' makes label for all particles.  Typically
        it should be 0, 1, 2 or perhaps 3.

    fig : a matplotlib figure instance, optional
        The figure canvas on which the plot will be drawn.

    ax : a matplotlib axis instance, optional
        The axis context in which the plot will be drawn.

    Returns
    -------
    fig, output : tuple
        A tuple of the matplotlib figure and the axes instance or animation
        instance used to produce the figure.

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

    if not isinstance(ket, list):
        kets = [ket]
    else:
        kets = ket

    _equal_shape(kets)

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

        artist = [ax.imshow(complex_array_to_rgb(qub, theme=theme),
                  interpolation="none",
                  extent=(0, size_x, 0, size_y))]
        artist_list.append(artist)

    if len(kets) == 1:
        output = ax
    else:
        output = animation.ArtistAnimation(fig, artist_list, interval=50,
                                           blit=True, repeat_delay=1000)

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

    return fig, output


def plot_schmidt(ket, theme='light', splitting=None,
                 labels_iteration=(3, 2), *, fig=None, ax=None):
    """
    Plotting scheme related to Schmidt decomposition.
    Converts a state into a matrix (A_ij -> A_i^j),
    where rows are first particles and columns - last.

    See also: plot_qubism with how='before_after' for a similar plot.

    .. note::

        colorblind_safe does not apply because of its unique colormap

    Parameters
    ----------
    ket : Qobj
        Pure state for plotting.

    theme : str {'light', 'dark'}, default: 'light'
        Set coloring theme for mapping complex values into colors.
        See: complex_array_to_rgb.

    splitting : int, optional
        Plot for a number of first particles versus the rest.
        If not given, it is (number of particles + 1) // 2.

    labels_iteration : int or pair of ints, default: (3, 2)
        Number of particles to be shown as tick labels,
        for first (vertical) and last (horizontal) particles, respectively.

    fig : a matplotlib figure instance, optional
        The figure canvas on which the plot will be drawn.

    ax : a matplotlib axis instance, optional
        The axis context in which the plot will be drawn.

    Returns
    -------
    fig, output : tuple
        A tuple of the matplotlib figure and the axes instance or animation
        instance used to produce the figure.

    """

    fig, ax = _is_fig_and_ax(fig, ax)

    if not isinstance(ket, list):
        kets = [ket]
    else:
        kets = ket

    _equal_shape(kets)

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

    if len(kets) == 1:
        output = ax
    else:
        output = animation.ArtistAnimation(fig, artist_list, interval=50,
                                           blit=True, repeat_delay=1000)

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

    return fig, output
