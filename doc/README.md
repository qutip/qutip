Repository for QuTiP documentation
==================================

This repository contains the source files for the QuTiP documentation.

For pre-built documentation, see https://www.qutip.org/documentation.html

Building
--------

The main Python requirements for the documentation are `sphinx`, `sphinx-gallery`, `sphinx_rtd_theme`, `numpydoc` and `ipython`.
You should build or install the version of QuTiP you want to build the documentation against in the same environment.
You will also need a sensible copy of `make`, and if you want to build the LaTeX documentation then also a `pdflatex` distribution.
As of 2021-04-20, the `conda` recipe for `sphinx_rtd_theme` is rather old compared to the `pip` version, so it's recommended to use a mostly `pip`-managed environment to do the documentation build.

The simplest way to get a functional build environment is to use the `requirements.txt` file in this repository, which completely defines a known-good `pip` environment (tested on Python 3.8, but not necessarily limited to it).
If you typically use conda, the way to do this is
```bash
$ conda create -n qutip-doc-build python=3.8
$ conda activate qutip-doc-build
$ pip install -r /path/to/qutip/doc/requirements.txt
```
You will also need to build or install the main QuTiP library in the same environment.
If you simply want to build the documentation without editing the main library, you can install a release version of QuTiP with `pip install qutip`.
Otherwise, refer to [the main repository](https://github.com/qutip/qutip) for the current process to build from source.
You need to have the optional QuTiP dependency `Cython` to build the documentation, but this is included in this repository's `requirements.txt` so you do not need to do anything separately.

After you have done this, you can effect the build with `make`.
The targets you might want are `html`, `latexpdf` and `clean`, which build the HTML pages, build the PDFs, and delete all built files respectively.
For example, to build the HTML files only, use
```bash
$ make html
```

*Note (2021-04-20):* the documentation build is currently broken on Windows due to incompatibilities in the main library in multiprocessing components.

Writing User Guides
-------------------

The user guide provides an overview of QuTiP's functionality. The guide is composed of individual reStructuredText (`.rst`) files which each get rendered as a webpage. Each page typically tackles one area of functionality. To learn more about how to write `.rst` files, it is useful to follow the [Sphinx Guide](https://www.sphinx-doc.org/en/master/usage/index.html).

The documentation build also utilizes a number of [Sphinx Extensions](https://www.sphinx-doc.org/en/master/usage/extensions/index.html) including but not limited to
[doctest](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html), [autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html), [sphinx gallery](https://sphinx-gallery.github.io/stable/index.html), [plot](https://matthew-brett.github.io/nb2plots/nbplots.html#module-nb2plots.nbplots). Additional extensions can be configured in the `conf.py` file.

Tests can also be run on examples in the documentation using the doctest extension
and plots are generated using the `plot` directive. For more specific
guidelines on how to incorporate code examples into the guide, refer to (insert reference).
