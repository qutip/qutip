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

In a conda environment use:
    
    $ conda install sphinx numpydoc sphinx_rtd_theme ipython

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
