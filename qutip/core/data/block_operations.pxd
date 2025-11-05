#cython: language_level=3

from qutip.core.data.csr cimport CSR
from qutip.core.data.dense cimport Dense
from qutip.core.data.base cimport Data, idxint

cpdef Dense block_build_dense(
    idxint[:] block_rows, idxint[:] block_cols, Data[:] blocks,
    idxint[:] block_widths, idxint[:] block_heights
)

cpdef CSR block_build_csr(
    idxint[:] block_rows, idxint[:] block_cols, Data[:] blocks,
    idxint[:] block_widths, idxint[:] block_heights
)

cpdef Dense block_extract_dense(
    Dense data,
    idxint row_start, idxint row_stop,
    idxint col_start, idxint col_stop
)

cpdef CSR block_extract_csr(
    CSR data,
    idxint row_start, idxint row_stop,
    idxint col_start, idxint col_stop
)

cpdef Dense block_overwrite_dense(
    Data data, Data block, idxint above, idxint before
)

cpdef CSR block_overwrite_csr(
    CSR data, CSR block, idxint above, idxint before
)