Repository for QuTiP documentation
==================================

This repository contains the source files for the QuTiP documentation.

For pre-built documentation, see http://www.qutip.org/documentation.html

Build requirements
------------------

* Sphinx: http://sphinx-doc.org/
* LaTeX and pdflatex.
* numpydoc
* ipython

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
