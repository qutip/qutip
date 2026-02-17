.. _settings:

**************
QuTiP settings
**************

QuTiP has multiple settings that control it's behaviour:

* ``qutip.settings`` contains installation and runtime information.
  Most of these parameters are readonly. But systems paths used by QuTiP are
  also included here and could need updating in none standard environment.
* ``qutip.settings.core`` contains options for operations with ``Qobj`` and
  other qutip's class. All options are writable.
* ``qutip.settings.compile`` has options that control compilation of string
  coefficients to cython modules. All options are writable.

.. _settings-install:

********************
Environment settings
********************

``qutip.settings`` has information about the run time environment:

.. tabularcolumns:: | p{3cm} | p{2cm} | p{10cm} |

.. cssclass:: table-striped

+-------------------+-----------+----------------------------------------------------------+
| Setting           | Read Only | Description                                              |
+===================+===========+==========================================================+
| `has_mkl`         | True      | Whether qutip can find mkl libraries.                    |
|                   |           | mkl sparse linear equation solver can be used when True. |
+-------------------+-----------+----------------------------------------------------------+
| `mkl_lib_location`| False     | Path of the mkl library.                                 |
+-------------------+-----------+----------------------------------------------------------+
| `mkl_lib`         | True      | Mkl libraries loaded with ctypes.                        |
+-------------------+-----------+----------------------------------------------------------+
| `ipython`         | True      | Whether running in IPython.                              |
+-------------------+-----------+----------------------------------------------------------+
| `eigh_unsafe`     | True      | When true, SciPy's `eigh` and `eigvalsh` are replaced    |
|                   |           | with custom implementations that call `eig` and          |
|                   |           | `eigvals` instead. This setting exists because in some   |
|                   |           | environments SciPy's `eigh` segfaults or gives invalid   |
|                   |           | results.                                                 |
+-------------------+-----------+----------------------------------------------------------+
| `coeffroot`       | False     | Directory in which QuTiP creates cython modules for      |
|                   |           | string coefficient.                                      |
+-------------------+-----------+----------------------------------------------------------+
| `coeff_write_ok`  | True      | Whether QuTiP has write permission for `coeffroot`.      |
+-------------------+-----------+----------------------------------------------------------+
| `idxint_size`     | True      | Whether QuTiP's sparse matrix indices use 32 or 64 bits. |
|                   |           | Sparse matrices' size are limited to 2**(idxint_size-1)  |
|                   |           | rows and columns.                                        |
+-------------------+-----------+----------------------------------------------------------+
| `num_cpus`        | True      | Detected number of cpus.                                 |
+-------------------+-----------+----------------------------------------------------------+
| `colorblind_safe` | False     | Control the default cmap in visualization functions.     |
+-------------------+-----------+----------------------------------------------------------+


It may be needed to update ``coeffroot`` if the default HOME is not writable. It can be done with:

>>> qutip.settings.coeffroot = "path/to/string/coeff/directory"

In QuTiP version 5 and later, strings compiled in a session are kept for future sessions.
As long as the same ``coeffroot`` is used, each string will only be compiled once.


*********************************
Modifying Internal QuTiP Settings
*********************************

.. _settings-params:

User Accessible Parameters
==========================

In this section we show how to modify a few of the internal parameters used by ``Qobj``.
The settings that can be modified are given in the following table:

.. tabularcolumns:: | p{3cm} | p{5cm} | p{5cm} |

.. cssclass:: table-striped

+------------------------------+----------------------------------------------+--------------------------------+
| Options                      | Description                                  | type [default]                 |
+==============================+==============================================+================================+
| `auto_tidyup`                | Automatically tidyup sparse quantum objects. | bool [True]                    |
+------------------------------+----------------------------------------------+--------------------------------+
| `auto_tidyup_atol`           | Tolerance used by tidyup. (sparse only)      | float [1e-14]                  |
+------------------------------+----------------------------------------------+--------------------------------+
| `auto_tidyup_dims`           | Whether the scalar dimension are contracted  | bool [False]                   |
+------------------------------+----------------------------------------------+--------------------------------+
| `atol`                       | General absolute tolerance.                  | float [1e-12]                  |
+------------------------------+----------------------------------------------+--------------------------------+
| `rtol`                       | General relative tolerance.                  | float [1e-12]                  |
+------------------------------+----------------------------------------------+--------------------------------+
| `function_coefficient_style` | Signature expected by function coefficients. | {["auto"], "pythonic", "dict"} |
+------------------------------+----------------------------------------------+--------------------------------+
| `default_dtype`              | Data format used when creating Qobj from     | {[None], "CSR", "Dense",       |
|                              | QuTiP functions, such as ``qeye``.           | "Dia"} + other from plugins    |
+------------------------------+----------------------------------------------+--------------------------------+

See also :class:`.CoreOptions`.

.. _settings-usage:

Example: Changing Settings
==========================

The two most important settings are ``auto_tidyup`` and ``auto_tidyup_atol`` as
they control whether the small elements of a quantum object should be removed,
and what number should be considered as the cut-off tolerance.
Modifying these, or any other parameters, is quite simple::

>>> qutip.settings.core["auto_tidyup"] = False

The settings can also be changed for a code block::

>>> with qutip.CoreOptions(atol=1e-5):
>>>     assert qutip.qeye(2) * 1e-9 == qutip.qzero(2)



.. _settings-compile:

String Coefficient Parameters
=============================

String based coefficient used for time dependent system are compiled using Cython when available.
Speeding the simulations, it tries to set c types to passed variables.
``qutip.settings.compile`` has multiple options for compilation.

There are options are about to whether to compile.

.. tabularcolumns:: | p{3cm} | p{10cm} |

.. cssclass:: table-striped

+--------------------------+-----------------------------------------------------------+
| Options                  | Description                                               |
+==========================+===========================================================+
| `use_cython`             | Whether to compile string using cython or using ``eval``. |
+--------------------------+-----------------------------------------------------------+
| `recompile`              | Whether to force recompilation or use a previously        |
|                          | constructed coefficient if available.                     |
+--------------------------+-----------------------------------------------------------+


Some options passed to cython and the compiler (for advanced user).

.. tabularcolumns:: | p{3cm} | p{10cm} |

.. cssclass:: table-striped

+--------------------------+-----------------------------------------------------------+
| Options                  | Description                                               |
+==========================+===========================================================+
| `compiler_flags`         | C++ compiler flags.                                       |
+--------------------------+-----------------------------------------------------------+
| `link_flags`             | C++ linker flags.                                         |
+--------------------------+-----------------------------------------------------------+
| `build_dir`              | cythonize's build_dir.                                    |
+--------------------------+-----------------------------------------------------------+
| `extra_import`           | import or cimport line of code to add to the cython file. |
+--------------------------+-----------------------------------------------------------+
| `clean_on_error`         | Whether to erase the created file if compilation failed.  |
+--------------------------+-----------------------------------------------------------+


Lastly some options control how qutip tries to detect C types (for advanced user).

.. tabularcolumns:: | p{3cm} | p{10cm} |

.. cssclass:: table-striped

+--------------------------+-----------------------------------------------------------------------------------------+
| Options                  | Description                                                                             |
+==========================+=========================================================================================+
| `try_parse`              | Whether QuTiP parses the string to detect common patterns.                              |
|                          |                                                                                         |
|                          | When True, "cos(w * t)" and "cos(a * t)" will use the same compiled coefficient.        |
+--------------------------+-----------------------------------------------------------------------------------------+
| `static_types`           | If False, every variable will be typed as ``object``, (except ``t`` which is double).   |
|                          |                                                                                         |
|                          | If True, scalar (int, float, complex), string and Data types are detected.              |
+--------------------------+-----------------------------------------------------------------------------------------+
| `accept_int`             | Whether to type ``args`` values which are Python ints as int or float/complex.          |
|                          |                                                                                         |
|                          | Per default it is True when subscription (``a[i]``) is used.                            |
+--------------------------+-----------------------------------------------------------------------------------------+
| `accept_float`           | Whether to type ``args`` values which are Python floats as int or float/complex.        |
|                          |                                                                                         |
|                          | Per default it is True when comparison (``a > b``) is used.                             |
+--------------------------+-----------------------------------------------------------------------------------------+


These options can be set at a global level in ``qutip.settings.compile`` or by passing a :class:`.CompilationOptions` instance to the :func:`.coefficient` functions.

>>> qutip.coefficient("cos(t)", compile_opt=CompilationOptions(recompile=True))
