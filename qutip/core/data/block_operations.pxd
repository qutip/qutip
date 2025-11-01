#cython: language_level=3

from qutip.core.data.csr cimport CSR
from qutip.core.data.dense cimport Dense
from qutip.core.data.base cimport Data, idxint

cpdef Dense concat_blocks_dense(
    idxint[:] block_rows, idxint[:] block_cols, Data[:] blocks,
    idxint[:] block_widths, idxint[:] block_heights
)

cpdef CSR concat_blocks_csr(
    idxint[:] block_rows, idxint[:] block_cols, Data[:] blocks,
    idxint[:] block_widths, idxint[:] block_heights
)

cpdef Dense slice_dense(
    Dense data,
    idxint row_start, idxint row_stop,
    idxint col_start, idxint col_stop
)

cpdef CSR slice_csr(
    CSR data,
    idxint row_start, idxint row_stop,
    idxint col_start, idxint col_stop
)

cpdef Dense insert_block_dense(
    Data data, Dense block, idxint above, idxint before
)

cpdef CSR insert_block_csr(
    CSR data, CSR block, idxint above, idxint before
)