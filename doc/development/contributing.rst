.. _development-contributing:

*********************************
Contributing to QuTiP Development
*********************************

Quick Start
===========

QuTiP is developed through wide collaboration using the ``git`` version-control system, with the main repositories hosted in the `qutip organisation on GitHub <https://github.com/qutip>`_.
You will need to be familiar with ``git`` as a tool, and the `GitHub Flow <https://docs.github.com/en/get-started/quickstart/github-flow>`_ workflow for branching and making pull requests.
The exact details of environment set-up, build process and testing vary by repository and are discussed below, however in overview, the steps to contribute are:

#. Consider creating an issue on the GitHub page of the relevant repository, describing the change you think should be made and why, so we can discuss details with you and make sure it is appropriate.
#. (If this is your first contribution.) Make a fork of the relevant repository on GitHub and clone it to your local computer.  Also add our copy as a remote (``git remote add qutip https://github.com/qutip/<repo>``)
#. Begin on the ``master`` branch (``git checkout master``), and pull in changes from the main QuTiP repository to make sure you have an up-to-date copy (``git pull qutip master``).
#. Switch to a new ``git`` branch (``git checkout -b <branch-name>``).
#. Make the changes you want to make, then create some commits with short, descriptive names (``git add <files>`` then ``git commit``).
#. Follow the build process for this repository to build the final result so you can check your changes work sensibly.
#. Run the tests for the repository (if it has them).
#. Push the changes to your fork (``git push -u origin <branch-name>``).  You won't be able to push to the main QuTiP repositories directly.
#. Go to the GitHub website for the repository you are contributing to, click on the "Pull Requests" tab, click the "New Pull Request" button, and follow the instructions there.

Once the pull request is created, some members of the QuTiP admin team will review the code to make sure it is suitable for inclusion in the library, to check the programming, and to ensure everything meets our standards.
For some repositories, several automated tests will run whenever you create or modify a pull request; in general these will be the same tests you can run locally, and all tests are required to pass online before your changes are merged.
There may be some feedback and possibly some requested changes.
You can add more commits to address these, and push them to the relevant branch of your fork to update the pull request.

The rest of this document covers programming standards, and particular considerations for some of the more complicated repositories.


.. _contributing-qutip:

Core Library: qutip/qutip
=========================

The core library is in the `qutip/qutip repository on GitHub <https://github.com/qutip/qutip>`_.

Building
--------

Building the core library from source is typically a bit more difficult than simply installing the package for regular use.
You will most likely want to do this in a clean Python environment so that you do not compromise a working installation of a release version, for example by starting from ::

   conda create -n qutip-dev python

:ref:`Complete instructions for the build <install>` are elsewhere in this guide, however beware that you will need to follow the :ref:`installation from source using setuptools section <build-setuptools>`, not the general installation.
You will need all the *build* and *tests* "optional" requirements for the package.
The build requirements can be found in the |pyproject.toml file|_, and the testing requirements are in the ``tests`` key of the ``options.extras_require`` section of |setup.cfg|_.
You will also need the requirements for any optional features you want to test as well.

.. |pyproject.toml file| replace:: ``pyproject.toml`` file
.. _pyproject.toml file: https://github.com/qutip/qutip/blob/master/pyproject.toml
.. |setup.cfg| replace:: ``setup.cfg``
.. _setup.cfg: https://github.com/qutip/qutip/blob/master/setup.cfg

Refer to the main instructions for the most up-to-date version, however as of version 4.6 the requirements can be installed into a conda environment with ::

   conda install setuptools wheel numpy scipy cython packaging pytest pytest-rerunfailures

Note that ``qutip`` should *not* be installed with ``conda install``.

.. note::
   If you prefer, you can also use ``pip`` to install all the dependencies.
   We typically recommend ``conda`` when doing main-library development because it is easier to switch low-level packages around like BLAS implementations, but if this doesn't mean anything to you, feel free to use ``pip``.

You will need to make sure you have a functioning C++ compiler to build QuTiP.
If you are on Linux or Mac, this is likely already done for you, however if you are on Windows, refer to the :ref:`Windows installation <install-on-windows>` section of the installation guide.

The command to build QuTiP in editable mode is ::

   python setup.py develop

from the repository directory.
If you now load up a Python interpreter, you should be able to ``import qutip`` from anywhere as long as the correct Python environment is active.
Any changes you make to the Python files in the git repository should be immediately present if you restart your Python interpreter and re-import ``qutip``.

On the first run, the setup command will compile many C++ extension modules built from Cython sources (files ending ``.pxd`` and ``.pyx``).
Generally the low-level linear algebra routines that QuTiP uses are written in these files, not in pure Python.
Unlike Python files, changes you make to Cython files will not appear until you run ``python setup.py develop`` again; you will only need to re-run this if you are changing Cython files.
Cython will detect and compile only the files that have been changed, so this command will be faster on subsequent runs.

.. note::

   When undertaking Cython development, the reason we use ``python setup.py develop`` instead of ``pip install -e .`` is because Cython's changed-file detection does not reliably work in the latter.
   ``pip`` tends to build in temporary virtual environments, which often makes Cython think its core library files have been updated, triggering a complete, slow rebuild of everything.

.. note::

    QuTiP follows `NEP29`_ when selecting the supported version of its dependencies.
    To see which versions are planned to be supported in the next release, please refer to the :ref:`release roadmap`.
    These coincide with the versions employed for testing in continuous integration.

    In the event of a feature requiring a version upgrade of python or a dependency, it will be considered appropriately in the pull request.
    In any case, python and dependency upgrades will only happen in mayor or minor versions of QuTiP, not in a patch.

.. _NEP29: https://numpy.org/neps/nep-0029-deprecation_policy.html


Code Style
----------

The biggest concern you should always have is to make it easy for your code to be read and understood by the person who comes next.

All new contributions must follow `PEP 8 style <https://peps.python.org/pep-0008/>`_; all pull requests will be passed through a linter that will complain if you violate it.
You should use the ``pycodestyle`` package locally (available on ``pip``) to test you satisfy the requirements before you push your commits, since this is rather faster than pushing 10 different commits trying to fix minor niggles.
Keep in mind that there is quite a lot of freedom in this style, especially when it comes to line breaks.
If a line is too long, consider the *best* way to split it up with the aim of making the code readable, not just the first thing that doesn't generate a warning.

Try to stay consistent with the style of the surrounding code.
This includes using the same variable names, especially if they are function arguments, even if these "break" PEP 8 guidelines.
*Do not* change existing parameter, attribute or method names to "match" PEP 8; these are breaking user-facing changes, and cannot be made except in a new major release of QuTiP.

Other than this, general "good-practice" Python standards apply: try not to duplicate code; try to keep functions short, descriptively-named and side-effect free; provide a docstring for every new function; and so on.

Type Hints
----------

Adding type hints to users facing functions is recommended.
QuTiP's approach is such:

- Type hints are *hints* for the users.
- Type hints can show the preferred usage over real implementation, for example:
  - ``Qobj.__mul__`` is typed to support product with scalar, not other ``Qobj``, for which ``__matmul__`` should is preferred.
  - ``solver.options`` claims it return a dict not ``_SolverOptions`` (which is a subclass of dict).
- Type alias are added to ``qutip.typing``.
- `Any` can be used for input which type can be extended by plugin modules, (``qutip-cupy``, ``qutip-jax``, etc.)


Documenting
-----------

When you make changes in the core library, you should update the relevant documentation if needed.
If you are making a bug fix, or other relatively minor changes, you will probably only need to make sure that the docstrings of the modified functions and classes are up-to-date; changes here will propagate through to the documentation the next time it is built.
Be sure to follow the |numpydoc|_ when writing docstrings.
All docstrings will be parsed as reStructuredText, and will form the API documentation section of the documentation.

.. |numpydoc| replace:: Numpy documentation standards (``numpydoc``)
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html

Testing
-------

We use ``pytest`` as our test runner.
The base way to run every test is ::

   pytest /path/to/repo/qutip/tests

This will take around 10 to 30 minutes, depending on your computer and how many of the optional requirements you have installed.
It is normal for some tests to be marked as "skip" or "xfail" in yellow; these are not problems.
True failures will appear in red and be called "fail" or "error".

While prototyping and making changes, you might want to use some of the filtering features of ``pytest``.
Instead of passing the whole ``tests`` directory to the ``pytest`` command, you can also pass a list of files.
You can also use the ``-k`` selector to only run tests whose names include a particular pattern, for example ::

   pytest qutip/tests/test_qobj.py -k "expm"

to run the tests of :meth:`Qobj.expm`.

Changelog Generation
--------------------

We use ``towncrier`` for tracking changes and generating a changelog.
When making a pull request, we require that you add a towncrier entry along with the code changes.
You should create a file named ``<PR number>.<change type>`` in the ``doc/changes`` directory, where the PR number should be substituted for ``<PR number>``, and ``<change type>`` is either ``feature``, ``bugfix``, ``doc``, ``removal``, ``misc``, or ``deprecation``,
depending on the type of change included in the PR.

You can also create this file by installing ``towncrier`` and running

   towncrier create <PR number>.<change type>

Running this will create a file in the ``doc/changes`` directory with a filename corresponding to the argument you passed to ``towncrier create``.
In this file, you should add a short description of the changes that the PR introduces.

.. _contributing-docs:

Documentation: qutip/qutip (doc directory)
==========================================

The core library is in the `qutip/qutip repository on GitHub, inside the doc directory <https://github.com/qutip/qutip>`_.

Building
--------

The documentation is built using ``sphinx``, ``matplotlib`` and ``numpydoc``, with several additional extensions including ``sphinx-gallery`` and ``sphinx-rtd-theme``.
The most up-to-date instructions and dependencies will be in the ``README.md`` file of the documentation directory.
You can see the rendered version of this file simply by going to the `documentation GitHub page <https://github.com/qutip/qutip/tree/master/doc>`_ and scrolling down.

Building the documentation can be a little finnicky on occasion.
You likely will want to keep a separate Python environment to build the documentation in, because some of the dependencies can have tight requirements that may conflict with your favourite tools for Python development.
We recommend creating an empty ``conda`` environment containing only Python with ::

   conda create -n qutip-doc python=3.8

and install all further dependencies with ``pip``.
There is a ``requirements.txt`` file in the repository root that fixes all package versions exactly into a known-good configuration for a completely empty environment, using ::

   pip install -r requirements.txt

This known-good configuration was intended for Python 3.8, though in principle it is possible that other Python versions will work.

.. note::

   We recommend you use ``pip`` to install dependencies for the documentation rather than ``conda`` because several necessary packages can be slower to update their ``conda`` recipes, so suitable versions may not be available.

The documentation build includes running many components of the main QuTiP library to generate figures and to test the output, and to generate all the API documentation.
You therefore need to have a version of QuTiP available in the same Python environment.
If you are only interested in updating the users' guide, you can use a release version of QuTiP, for example by running ``pip install qutip``.
If you are also modifying the main library, you need to make your development version accessible in this environment.
See the `above section on building QuTiP <contributing-qutip_>`_ for more details, though the ``requirements.txt`` file will have already installed all the build requirements, so you should be able to simply run ::

   python setup.py develop

in the main library repository.

The documentation is built by running the ``make`` command.
There are several targets to build, but the most useful will be ``html`` to build the webpage documentation, ``latexpdf`` to build the PDF documentation (you will also need a full ``pdflatex`` installation), and ``clean`` to remove all built files.
The most important command you will want to run is ::

   make html

You should re-run this any time you make changes, and it should only update files that have been changed.

.. important::
   The documentation build includes running almost all the optional features of QuTiP.
   If you get failure messages in red, make sure you have installed all of the optional dependencies for the main library.

The HTML files will be placed in the ``_build/html`` directory.
You can open the file ``_build/html/index.html`` in your web browser to check the output.

Code Style
----------

All user guide pages and docstrings are parsed by Sphinx using reStructuredText.
There is a general `Sphinx usage guide <https://www.sphinx-doc.org/en/master/usage/index.html>`_, which has a lot of information that can sometimes be a little tricky to follow.
It may be easier just to look at other ``.rst`` files already in the documentation to copy the different styles.

.. note::
   reStructuredText is a very different language to the Markdown that you might be familiar with.
   It's always worth checking your work in a web browser to make sure it's appeared the way you intended.

Testing
-------

There are unfortunately no automated tests for the documentation.
You should ensure that no errors appeared in red when you ran ``make html``.
Try not to introduce any new warnings during the build process.
The main test is to open the HTML pages you have built (open ``_build/html/index.html`` in your web browser), and click through to the relevant pages to make sure everything has rendered the way you expected it to.
