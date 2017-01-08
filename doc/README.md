Repository for QuTiP documentation
==================================

This repository contains the source files for the QuTiP documentation.

For pre-built documentation, see http://www.qutip.org/documentation.html

Build requirements
------------------

* Sphinx: http://sphinx-doc.org/
* sphinx_rtd_theme
* LaTeX and pdflatex.
* numpydoc
* ipython

2017-01-07: Partially successful building using:

* sphinx v1.5.1
* numpydoc v0.6.0
* ipython 5.1.0

Build
-----

To build the documentation on Linux or OS X run:

    $ make html latexpdf

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
