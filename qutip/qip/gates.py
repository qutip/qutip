import warnings

from qutip.qip.operations.gates import *
warnings.warn(
    "Importation from qutip.qip.gates is deprecated."
    "Please use e.g.\n from qutip.qip.operations import cnot\n",
    DeprecationWarning, stacklevel=2)
