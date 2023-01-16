.. _settings:

*********************************
Modifying Internal QuTiP Settings
*********************************

.. _settings-params:

User Accessible Parameters
==========================

In this section we show how to modify a few of the internal parameters used by QuTiP.
The settings that can be modified are given in the following table:

.. tabularcolumns:: | p{3cm} | p{5cm} | p{5cm} |

.. cssclass:: table-striped

+------------------------------+----------------------------------------------+------------------------------+
| Setting                      | Description                                  | Options                      |
+==============================+==============================================+==============================+
| `auto_tidyup`                | Automatically tidyup sparse quantum objects. | True / False                 |
+------------------------------+----------------------------------------------+------------------------------+
| `auto_tidyup_atol`           | Tolerance used by tidyup. (sparse only)      | float {1e-14}                |
+------------------------------+----------------------------------------------+------------------------------+
| `atol`                       | General absolute tolerance.                  | float {1e-12}                |
+------------------------------+----------------------------------------------+------------------------------+
| `rtol`                       | General relative tolerance.                  | float {1e-12}                |
+------------------------------+----------------------------------------------+------------------------------+
| `function_coefficient_style` | Signature expected by function coefficients. | {"auto", "pythonic", "dict"} |
+------------------------------+----------------------------------------------+------------------------------+

.. _settings-usage:

Example: Changing Settings
==========================

The two most important settings are ``auto_tidyup`` and ``auto_tidyup_atol`` as they control whether the small elements of a quantum object should be removed, and what number should be considered as the cut-off tolerance.
Modifying these, or any other parameters, is quite simple::

>>> qutip.settings.core["auto_tidyup"] = False

The settings can also be changed for a code block::

>>> with qutip.CoreOptions(atol=1e-5):
>>>     assert qutip.qeye(2) * 1e-9 == qutip.qzero(2)
