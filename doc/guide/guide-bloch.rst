.. _bloch:

******************************
Plotting on the Bloch Sphere
******************************

.. _bloch-intro:

Introduction
============

When studying the dynamics of a two-level system, it is often convenient to visualize the state of the system by plotting the state-vector or density matrix on the Bloch sphere.  In QuTiP, we have created two different classes to allow for easy creation and manipulation of data sets, both vectors and data points, on the Bloch sphere.  The :class:`qutip.Bloch` class, uses Matplotlib to render the Bloch sphere, where as :class:`qutip.Bloch3d` uses the Mayavi rendering engine to generate a more faithful 3D reconstruction of the Bloch sphere.

.. _bloch-class:

The Bloch and Bloch3d Classes
=============================

In QuTiP, creating a Bloch sphere is accomplished by calling either:

.. plot::
    :context:

    b = qutip.Bloch()

which will load an instance of the :class:`qutip.Bloch` class, or using ::

   >>> b3d = qutip.Bloch3d()

that loads the :class:`qutip.Bloch3d` version.  Before getting into the details of these objects, we can simply plot the blank Bloch sphere associated with these instances via:

.. plot::
    :context:

    b.make_sphere()

or

.. _image-blank3d:

.. figure:: figures/bloch3d-blank.png
    :width: 3.5in
    :figclass: align-center

In addition to the ``show`` command, see the API documentation for :class:`~Bloch` for a full list of other available functions.
As an example, we can add a single data point:

.. plot::
    :context: close-figs

    pnt = [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
    b.add_points(pnt)
    b.render()

and then a single vector:

.. plot::
    :context: close-figs

    b.fig.clf()
    vec = [0, 1, 0]
    b.add_vectors(vec)
    b.render()

and then add another vector corresponding to the :math:`\left|\rm up \right>` state:

.. plot::
    :context: close-figs

    up = qutip.basis(2, 0)
    b.add_states(up)
    b.render()

Notice that when we add more than a single vector (or data point), a different color will automatically be applied to the later data set (mod 4).
In total, the code for constructing our Bloch sphere with one vector, one state, and a single data point is:

.. plot::
    :context: close-figs

    b = qutip.Bloch()

    pnt = [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)]
    b.add_points(pnt)
    vec = [0, 1, 0]
    b.add_vectors(vec)
    up = qutip.basis(2, 0)
    b.add_states(up)
    b.render()

where we have removed the extra ``show()`` commands.  Replacing ``b=Bloch()`` with ``b=Bloch3d()`` in the above code generates the following 3D Bloch sphere.

.. _image-bloch3ddata:

.. figure:: figures/bloch3d+data.png
    :width: 3.5in
    :figclass: align-center


We can also plot multiple points, vectors, and states at the same time by passing list or arrays instead of individual elements.  Before giving an example, we can use the `clear()` command to remove the current data from our Bloch sphere instead of creating a new instance:

.. plot::
    :context: close-figs

    b.clear()
    b.render()


Now on the same Bloch sphere, we can plot the three states associated with the x, y, and z directions:

.. plot::
    :context: close-figs

    x = (qutip.basis(2, 0) + (1+0j)*qutip.basis(2, 1)).unit()
    y = (qutip.basis(2, 0) + (0+1j)*qutip.basis(2, 1)).unit()
    z = (qutip.basis(2, 0) + (0+0j)*qutip.basis(2, 1)).unit()

    b.add_states([x, y, z])
    b.render()

a similar method works for adding vectors:

.. plot::
    :context: close-figs

    b.clear()
    vec = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    b.add_vectors(vec)
    b.render()

Adding multiple points to the Bloch sphere works slightly differently than adding multiple states or vectors.  For example, lets add a set of 20 points around the equator (after calling `clear()`):

.. plot::
    :context: close-figs

    b.clear()

    th = np.linspace(0, 2*np.pi, 20)
    xp = np.cos(th)
    yp = np.sin(th)
    zp = np.zeros(20)

    pnts = [xp, yp, zp]
    b.add_points(pnts)
    b.render()

Notice that, in contrast to states or vectors, each point remains the same color as the initial point.  This is because adding multiple data points using the ``add_points`` function is interpreted, by default, to correspond to a single data point (single qubit state) plotted at different times.  This is very useful when visualizing the dynamics of a qubit.  An example of this is given in the example .  If we want to plot additional qubit states we can call additional ``add_points`` functions:

.. plot::
    :context: close-figs

    xz = np.zeros(20)
    yz = np.sin(th)
    zz = np.cos(th)
    b.add_points([xz, yz, zz])
    b.render()

The color and shape of the data points is varied automatically by the Bloch class.  Notice how the color and point markers change for each set of data.  Again, we have had to call ``add_points`` twice because adding more than one set of multiple data points is *not* supported by the ``add_points`` function.

What if we want to vary the color of our points.  We can tell the :class:`qutip.Bloch` class to vary the color of each point according to the colors listed in the ``b.point_color`` list (see :ref:`bloch-config` below).  Again after ``clear()``:

.. plot::
    :context: close-figs

    b.clear()

    xp = np.cos(th)
    yp = np.sin(th)
    zp = np.zeros(20)
    pnts = [xp, yp, zp]
    b.add_points(pnts, 'm')  # <-- add a 'm' string to signify 'multi' colored points
    b.render()


Now, the data points cycle through a variety of predefined colors.  Now lets add another set of points, but this time we want the set to be a single color, representing say a qubit going from the :math:`\left|\rm up\right>` state to the :math:`\left|\rm down\right>` state in the y-z plane:

.. plot::
    :context: close-figs

    xz = np.zeros(20)
    yz = np.sin(th)
    zz = np.cos(th)

    b.add_points([xz, yz, zz])  # no 'm'
    b.render()

Again, the same plot can be generated using the :class:`qutip.Bloch3d` class by replacing ``Bloch`` with ``Bloch3d``:

.. figure:: figures/bloch3d+points.png
    :width: 3.5in
    :figclass: align-center

A more slick way of using this 'multi' color feature is also given in the example, where we set the color of the markers as a function of time.

Differences Between Bloch and Bloch3d
-------------------------------------
While in general the ``Bloch`` and ``Bloch3d`` classes are interchangeable, there are some important differences to consider when choosing between them.

- The ``Bloch`` class uses Matplotlib to generate figures.  As such, the data plotted on the sphere is in reality just a 2D object.  In contrast the ``Bloch3d`` class uses the 3D rendering engine from VTK via mayavi to generate the sphere and the included data.  In this sense the ``Bloch3d`` class is much more advanced, as objects are rendered in 3D leading to a higher quality figure.

- Only the ``Bloch`` class can be embedded in a Matplotlib figure window.  Thus if you want to combine a Bloch sphere with another figure generated in QuTiP, you can not use ``Bloch3d``.  Of course you can always post-process your figures using other software to get the desired result.

- Due to limitations in the rendering engine, the ``Bloch3d`` class does not support LaTeX for text.  Again, you can get around this by post-processing.

- The user customizable attributes for the ``Bloch`` and ``Bloch3d`` classes are not identical.  Therefore, if you change the properties of one of the classes, these changes will cause an exception if the class is switched.


.. _bloch-config:

Configuring the Bloch sphere
============================

Bloch Class Options
--------------------

At the end of the last section we saw that the colors and marker shapes of the data plotted on the Bloch sphere are automatically varied according to the number of points and vectors added.  But what if you want a different choice of color, or you want your sphere to be purple with different axes labels? Well then you are in luck as the Bloch class has 22 attributes which one can control.  Assuming ``b=Bloch()``:

.. tabularcolumns:: | p{3cm} | p{7cm} |  p{7cm} |

.. cssclass:: table-striped

+---------------+---------------------------------------------------------+-------------------------------------------------+
| Attribute     | Function                                                | Default Setting                                 |
+===============+=========================================================+=================================================+
| b.axes        | Matplotlib axes instance for animations. Set by ``axes``| ``None``                                        |
|               | keyword arg.                                            |                                                 |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.fig         | User supplied Matplotlib Figure instance. Set by ``fig``| ``None``                                        |
|               | keyword arg.                                            |                                                 |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.font_color  | Color of fonts                                          | 'black'                                         |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.font_size   | Size of fonts                                           | 20                                              |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.frame_alpha | Transparency of wireframe                               | 0.1                                             |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.frame_color | Color of wireframe                                      | 'gray'                                          |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.frame_width | Width of wireframe                                      | 1                                               |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.point_color | List of colors for Bloch point markers to cycle through | ``['b', 'r', 'g', '#CC6600']``                  |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.point_marker| List of point marker shapes to cycle through            | ``['o', 's', 'd', '^']``                        |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.point_size  | List of point marker sizes (not all markers look the    | ``[55, 62, 65, 75]``                            |
|               | same size when plotted)                                 |                                                 |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.sphere_alpha| Transparency of Bloch sphere                            | 0.2                                             |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.sphere_color| Color of Bloch sphere                                   | ``'#FFDDDD'``                                   |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.size        | Sets size of figure window                              | ``[7, 7]`` (700x700 pixels)                     |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.vector_color| List of colors for Bloch vectors to cycle through       | ``['g', '#CC6600', 'b', 'r']``                  |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.vector_width| Width of Bloch vectors                                  | 4                                               |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.view        | Azimuthal and Elevation viewing angles                  | ``[-60,30]``                                    |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.xlabel      | Labels for x-axis                                       | ``['$x$', '']`` +x and -x (labels use LaTeX)    |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.xlpos       | Position of x-axis labels                               | ``[1.1, -1.1]``                                 |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.ylabel      | Labels for y-axis                                       | ``['$y$', '']`` +y and -y (labels use LaTeX)    |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.ylpos       | Position of y-axis labels                               | ``[1.2, -1.2]``                                 |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.zlabel      | Labels for z-axis                                       | ``['$\left|0\right>$', '$\left|1\right>$']``    |
|               |                                                         | +z and -z (labels use LaTeX)                    |
+---------------+---------------------------------------------------------+-------------------------------------------------+
| b.zlpos       | Position of z-axis labels                               | ``[1.2, -1.2]``                                 |
+---------------+---------------------------------------------------------+-------------------------------------------------+

Bloch3d Class Options
---------------------

The Bloch3d sphere is also customizable.  Note however that the attributes for the ``Bloch3d`` class are not in one-to-one
correspondence to those of the ``Bloch`` class due to the different underlying rendering engines. Assuming ``b=Bloch3d()``:

.. tabularcolumns:: | p{3cm} | p{7cm} |  p{7cm} |

.. cssclass:: table-striped

+---------------+---------------------------------------------------------+---------------------------------------------+
| Attribute     | Function                                                | Default Setting                             |
+===============+=========================================================+=============================================+
| b.fig         | User supplied Mayavi Figure instance. Set by ``fig``    | ``None``                                    |
|               | keyword arg.                                            |                                             |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.font_color  | Color of fonts                                          | ``'black'``                                 |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.font_scale  | Scale of fonts                                          | 0.08                                        |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.frame       | Draw wireframe for sphere?                              | ``True``                                    |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.frame_alpha | Transparency of wireframe                               | 0.05                                        |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.frame_color | Color of wireframe                                      | ``'gray'``                                  |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.frame_num   | Number of wireframe elements to draw                    | 8                                           |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.frame_radius| Radius of wireframe lines                               | 0.005                                       |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.point_color | List of colors for Bloch point markers to cycle through | ``['r', 'g', 'b', 'y']``                    |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.point_mode  | Type of point markers to draw                           | ``'sphere'``                                |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.point_size  | Size of points                                          | 0.075                                       |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.sphere_alpha| Transparency of Bloch sphere                            | 0.1                                         |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.sphere_color| Color of Bloch sphere                                   | ``'#808080'``                               |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.size        | Sets size of figure window                              | ``[500, 500]`` (500x500 pixels)             |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.vector_color| List of colors for Bloch vectors to cycle through       | ``['r', 'g', 'b', 'y']``                    |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.vector_width| Width of Bloch vectors                                  | 3                                           |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.view        | Azimuthal and Elevation viewing angles                  | ``[45, 65]``                                |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.xlabel      | Labels for x-axis                                       | ``['|x>', '']`` +x and -x                   |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.xlpos       | Position of x-axis labels                               | ``[1.07, -1.07]``                           |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.ylabel      | Labels for y-axis                                       | ``['$y$', '']`` +y and -y                   |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.ylpos       | Position of y-axis labels                               | ``[1.07, -1.07]``                           |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.zlabel      | Labels for z-axis                                       | ``['|0>', '|1>']`` +z and -z                |
+---------------+---------------------------------------------------------+---------------------------------------------+
| b.zlpos       | Position of z-axis labels                               | ``[1.07, -1.07]``                           |
+---------------+---------------------------------------------------------+---------------------------------------------+

These properties can also be accessed via the print command:

.. doctest::

    >>> b = qutip.Bloch()

    >>> print(b) # doctest: +NORMALIZE_WHITESPACE
    Bloch data:
    -----------
    Number of points:  0
    Number of vectors: 0
    <BLANKLINE>
    Bloch sphere properties:
    ------------------------
    font_color:      black
    font_size:       20
    frame_alpha:     0.2
    frame_color:     gray
    frame_width:     1
    point_color:     ['b', 'r', 'g', '#CC6600']
    point_marker:    ['o', 's', 'd', '^']
    point_size:      [25, 32, 35, 45]
    sphere_alpha:    0.2
    sphere_color:    #FFDDDD
    figsize:         [5, 5]
    vector_color:    ['g', '#CC6600', 'b', 'r']
    vector_width:    3
    vector_style:    -|>
    vector_mutation: 20
    view:            [-60, 30]
    xlabel:          ['$x$', '']
    xlpos:           [1.2, -1.2]
    ylabel:          ['$y$', '']
    ylpos:           [1.2, -1.2]
    zlabel:          ['$\\left|0\\right>$', '$\\left|1\\right>$']
    zlpos:           [1.2, -1.2]
    <BLANKLINE>

.. _bloch-animate:

Animating with the Bloch sphere
===============================

The Bloch class was designed from the outset to generate animations.  To animate a set of vectors or data points the basic idea is: plot the data at time t1, save the sphere, clear the sphere, plot data at t2,... The Bloch sphere will automatically number the output file based on how many times the object has been saved (this is stored in b.savenum).  The easiest way to animate data on the Bloch sphere is to use the ``save()`` method and generate a series of images to convert into an animation.  However, as of Matplotlib version 1.1, creating animations is built-in.  We will demonstrate both methods by looking at the decay of a qubit on the bloch sphere.

.. _bloch-animate-decay:

Example: Qubit Decay
--------------------

The code for calculating the expectation values for the Pauli spin operators of a qubit decay is given below.  This code is common to both animation examples.

.. literalinclude:: scripts/ex_bloch_animation.py

.. _bloch-animate-decay-images:

Generating Images for Animation
++++++++++++++++++++++++++++++++

An example of generating images for generating an animation outside of Python is given below::

     import numpy as np
     b = qutip.Bloch()
     b.vector_color = ['r']
     b.view = [-40, 30]
     for i in range(len(sx)):
         b.clear()
         b.add_vectors([np.sin(theta), 0, np.cos(theta)])
         b.add_points([sx[:i+1], sy[:i+1], sz[:i+1]])
         b.save(dirc='temp')  # saving images to temp directory in current working directory

Generating an animation using FFmpeg (for example) is fairly simple::

   ffmpeg -i temp/bloch_%01d.png bloch.mp4

.. _bloch-animate-decay-direct:

Directly Generating an Animation
++++++++++++++++++++++++++++++++

.. important::
   Generating animations directly from Matplotlib requires installing either MEncoder or FFmpeg.
   While either choice works on linux, it is best to choose FFmpeg when running on the Mac.
   If using macports just do: ``sudo port install ffmpeg``.

The code to directly generate an mp4 movie of the Qubit decay is as follows ::

   from matplotlib import pyplot, animation
   from mpl_toolkits.mplot3d import Axes3D

   fig = pyplot.figure()
   ax = Axes3D(fig, azim=-40, elev=30)
   sphere = qutip.Bloch(axes=ax)

   def animate(i):
      sphere.clear()
      sphere.add_vectors([np.sin(theta), 0, np.cos(theta)])
      sphere.add_points([sx[:i+1], sy[:i+1], sz[:i+1]])
      sphere.make_sphere()
      return ax

   def init():
      sphere.vector_color = ['r']
      return ax

   ani = animation.FuncAnimation(fig, animate, np.arange(len(sx)),
                                 init_func=init, blit=False, repeat=False)
   ani.save('bloch_sphere.mp4', fps=20)

The resulting movie may be viewed here: `bloch_decay.mp4 <https://raw.githubusercontent.com/qutip/qutip/master/doc/figures/bloch_decay.mp4>`_
