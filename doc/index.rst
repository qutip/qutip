.. figure:: figures/logo.png
   :align: center
   :width: 7in


QuTiP: Quantum Toolbox in Python
================================


This documentation contains a user guide and automatically generated API documentation for QuTiP.
For more information see the `QuTiP project web page <https://qutip.org/>`_.
Here, you can also find a collection of `tutorials for QuTiP <https://qutip.org/qutip-tutorials/>`_.


.. toctree::
   :maxdepth: 3

   frontmatter.rst
   installation.rst
   guide/guide.rst
   apidoc/apidoc.rst

   changelog.rst
   contributors.rst
   development/development.rst
   advanced/64bit_indices  
   biblio.rst
   copyright.rst


Indices and tables
====================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Building QuTiP with 64-bit Integer Indices
==========================================
By default, QuTiP uses 32-bit integer indices. However, you can build QuTiP with 
64-bit integer indices by specifying a configuration option during installation.

To enable 64-bit integer indices, use the following command when building QuTiP:

.. code-block:: bash

    python -m build \
        --wheel \
        --config-setting="--global-option=--with-idxint-64"

This ensures that QuTiP uses 64-bit indices, which may be useful for very large 
quantum systems.


