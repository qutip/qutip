.. _environments guide:

************************************
Environments of Open Quantum Systems
************************************

*written by* |pm|_ *and* |gs|_

.. _pm: https://www.menczel.net/
.. |pm| replace:: *Paul Menczel*
.. _gs: https://gsuarezr.github.io/
.. |gs| replace:: *Gerardo Suarez*
.. (this is a workaround for italic links in rst)

QuTiP can describe environments of open quantum systems.
They can be passed to various solvers, where their influence is taken into account exactly or approximately.
In the following, we will discuss bosonic and fermionic thermal environments.
In our definitions, we follow [BoFiN23]_.

Note that currently, we only support a single coupling term per environment.
If a more generalized coupling would be useful to you, please let us know on GitHub.


.. toctree::
   :maxdepth: 2

   environments/bosonic.rst
   environments/fermionic.rst
   environments/approximations.rst
