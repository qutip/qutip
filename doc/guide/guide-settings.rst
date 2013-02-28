.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _settings:

*********************************
Modifying Internal QuTiP Settings
*********************************

.. _settings-params:

User Accessible Parameters
==========================

.. note:: This section deals with modifying the internal QuTiP settings.  Use only if you know what you are doing.

In this section we show how to modify a few of the internal parameters used by QuTiP.  The settings that can be modified are given in the following table:

.. tabularcolumns:: | p{3cm} | p{5cm} | p{5cm} |

+-------------------------------+-------------------------------------------+-----------------------------+
| Setting                       | Description                               | Options                     |
+===============================+===========================================+=============================+
| `qutip_graphics`              | Use matplotlib                            | True / False                |
+-------------------------------+-------------------------------------------+-----------------------------+
| `qutip_gui`                   | Pick GUI library, or disable.             | "PYSIDE","PYQT4","NONE"     |
+-------------------------------+-------------------------------------------+-----------------------------+
| `auto_herm`                   | Automatically calculate the hermicity of  | True / False                |
|                               | quantum objects.                          |                             |
+-------------------------------+-------------------------------------------+-----------------------------+
| `auto_tidyup`                 | Automatically tidyup quantum objects.     | True / False                |
+-------------------------------+-------------------------------------------+-----------------------------+
| `auto_tidyup_atol`            | Tolerance used by tidyup                  | any `float` value > 0       |
+-------------------------------+-------------------------------------------+-----------------------------+
| `num_cpus`                    | Number of CPU's used for multiprocessing. | `int` between 1 and # cpu's |
+-------------------------------+-------------------------------------------+-----------------------------+
| `debug`                       | Show debug printouts.                     | True / False                |
+-------------------------------+-------------------------------------------+-----------------------------+

.. _settings-usage:

Example: Changing Settings
==========================

The two most important settings are ``auto_tidyup`` and ``auto_tidyup_atol`` as they control whether the small elements of a quantum object should be removed, and what number should be considered as the cut-off tolerance. Modifying these, or any other parameters, is quite simple::

>>> qutip.settings.auto_tidyup = False
>>> qutip.settings.qutip_gui = "NONE"

These settings will be used for the current QuTiP session only and will need to be modified again when restarting QuTiP.  If running QuTiP from a script file, then place the `qutip.setings.xxxx` commands immediately after `from qutip import *` at the top of the script file.  If you want to reset the parameters back to their default values then call the reset command::

>>> qutip.settings.reset()

Persistent Settings
===================

When QuTiP is imported, it looks for the file ``.qutiprc`` in the user's home directory. If this file is found, it will be loaded and overwrite the QuTiP default settings, which allows for persistent changes in the QuTiP settings to be made. A sample ``.qutiprc`` file is show below. The syntax is a simple key-value format, where the keys and possible values are described in the table above::

    # QuTiP Graphics
    qutip_graphics="YES"
    # QuTiP GUI selection
    qutip_gui=PYSIDE
    # use auto tidyup
    auto_tidyup=True
    # detect hermiticity
    auto_herm=True
    # use auto tidyup absolute tolerance 
    auto_tidyup_atol=1e-12
    # number of cpus
    num_cpus=4
    # debug
    debug=False


