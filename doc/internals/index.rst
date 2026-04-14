QuTiP's internals
=================

This section documents some of the internals of the QuTiP code base, in particular, the data layer that allows quantum objects to be stored in different representations internally. It is intended for people who are looking to modify and
extend QuTiP, particularly the lower-level, more heavily optimised components. See also the general :ref:`development` in this guide.

In general we assume that you have a working knowledge of using QuTiP as a user,
and these pages will not contain a tutorial on how to work with the library.  If
that is what you are looking for, please refer to the :ref:`guide` or the large
collection of `example notebooks`_.

.. _GitHub repository qutip/qutip: https://github.com/qutip/qutip
.. _user guide: http://qutip.org/docs/latest/
.. _example notebooks: http://qutip.org/tutorials.html

.. toctree::
   :maxdepth: 2
   :caption: Guide Contents:

   quantum-objects
   data-layer/index
