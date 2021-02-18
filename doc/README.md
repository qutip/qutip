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

In a conda environment use:

    $ conda install sphinx numpydoc sphinx_rtd_theme sphinx-gallery ipython

2017-03-28: Successful building using:

* sphinx v1.5.1
* numpydoc v0.6.0
* ipython 5.1.0

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

Building the HTML documentation requires Sphinx and numpydoc, both of which can be installed using Anaconda:

    > conda install sphinx numpydoc

If you installed QuTiP using another distribution of Python, these dependencies can also be installed using either ``easy_install`` or ``pip``:

    > easy_install install sphinx numpydoc
    > pip install sphinx numpydoc

To build the HTML documentation on Windows using ``cmd.exe``, run:

    > make html

From PowerShell, run:

    PS> .\make html

Writing User Guides
-------------------

The user guide provides an overview of QuTiP's functionality. The guide is composed of individual reStructuredText (`.rst`) files which each get rendered as a webpage. Each page typically tackles one area of functionality. To learn more about how to write `.rst` files, it is useful to follow the [Sphinx Guide](https://www.sphinx-doc.org/en/master/usage/index.html).

The documentation build also utilizes a number of [Sphinx Extensions](https://www.sphinx-doc.org/en/master/usage/extensions/index.html) including but not limited to
[doctest](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html) , [autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) , [sphinx gallery](https://sphinx-gallery.github.io/stable/index.html) , [plot](http://matthew-brett.github.io/nb2plots/nbplots.html#module-nb2plots.nbplots) . Additional extensions can be configured in the `conf.py` file.

Tests can also be run on examples in the documentation using the doctest extension
and plots are generated using the `plot` directive. For more specific
guidelines on how to incorporate code examples into the guide, refer to (insert reference).
