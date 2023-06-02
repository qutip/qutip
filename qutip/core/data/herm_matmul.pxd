#cython: language_level=3
from qutip.core.data.base cimport idxint, Data
from qutip.core.data.dense cimport Dense
from qutip.core.data.csr cimport CSR

cpdef Dense herm_matmul_csr_dense_dense(
    CSR left, Dense right,
    idxint subsystem_size=*, double complex scale=*,
    Data out=*
)

cpdef Dense herm_matmul_dense(
    Dense left, Dense right,
    idxint subsystem_size=*, double complex scale=*,
    Data out=*
)

cpdef Data herm_matmul_data(
    Data left, Data right,
    size_t subsystem_size=*, double complex scale=*,
    Data out=*
)
