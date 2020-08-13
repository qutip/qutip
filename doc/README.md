Repository for QuTiP documentation
==================================

This repository contains the source files for the QuTiP documentation.

For pre-built documentation, see http://www.qutip.org/documentation.html

Build requirements
------------------

* Sphinx: http://sphinx-doc.org/
* sphinx-gallery
* sphinx_rtd_theme
* LaTeX and pdflatex.
* numpydoc
* ipython
* nb2plots

In a conda environment use:

    $ conda install sphinx numpydoc sphinx_rtd_theme sphinx-gallery ipython nb2plots

2017-03-28: Successful building using:

* sphinx v1.5.1
* numpydoc v0.6.0
* ipython 5.1.0
* nb2plots 0.6

Build
-----
2017-01-07:
Thanks to some bug in ipython/ipython#8733 to do with the `ipython_savefig_dir` conf option,
then note that this build directory structure must exist already:

    _build/html/_images
    _build/latex/_images

To build the documentation on Linux or OS X run:

    $ make html latexpdf

To run doctest:

    $ make doctest

Building Documentation On Windows
---------------------------------

Building the HTML documentation requires Sphinx, numpydoc and nb2plots, both of which can be installed using Anaconda:

    > conda install sphinx numpydoc nb2plots

If you installed QuTiP using another distribution of Python, these dependencies can also be installed using either ``easy_install`` or ``pip``:

    > easy_install install sphinx numpydoc nb2plots
    > pip install sphinx numpydoc nb2plots

To build the HTML documentation on Windows using ``cmd.exe``, run:

    > make html

From PowerShell, run:

    PS> .\make html

Writing User Guides
-------------------

The user guide provides an overview of QuTiP's functionality. The guide is composed of individual reStructuredText (`.rst`) files which each get rendered as a webpage. Each page typically tackles one area of functionality. To learn more about how to write `.rst` files, it is useful to follow the `Sphinx Guide <https://www.sphinx-doc.org/en/master/usage/index.html>`_ .

The documentation build also utilizes a number of `Sphinx Extensions <https://www.sphinx-doc.org/en/master/usage/extensions/index.html>`_ including but not limited to
`doctest <https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html>`_ , `autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_ , `sphinx gallery <https://sphinx-gallery.github.io/stable/index.html>`_ , `nb2plots <http://matthew-brett.github.io/nb2plots/nbplots.html#module-nb2plots.nbplots>`_ . Additional extensions can be configured in the `conf.py` file.

Tests can also be run on examples in the documentation using the doctest extension
and plots are generated using the `plot` or `nbplot` directive. For more specific
guidelines on how to incorporate code examples into the guide, refer to (insert reference). 
