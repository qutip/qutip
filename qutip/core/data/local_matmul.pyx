#cython: language_level=3

from qutip.core.data cimport Data, Dense, dense, Dia
from qutip.core.data.matmul cimport imatmul_data_dense, matmul_dense
from qutip.core.data.matmul import matmul, matmul_dense_dia_dense
from qutip.core.data.add cimport iadd_dense
import numpy as np


#TODO:
# - One head function
# - Reusable interne functions + factory
# - What about jax etc?
# - merge left and right?
