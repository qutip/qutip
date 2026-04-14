#ifndef MATMUL_CSR_DENSE_HPP
#define MATMUL_CSR_DENSE_HPP

#include <complex>

#if defined(__GNUC__) || defined(_MSC_VER)
# define _RESTRICT __restrict
#else
# define _RESTRICT
#endif

/**
 * Compute out += scale * (csr @ dense) for C-ordered dense matrices.
 * 
 * For each row of the sparse matrix, we iterate over its non-zero elements
 * and accumulate: out[row, :] += scale * csr[row, col_idx] * dense[col_idx, :]
 * 
 * Parameters:
 *   csr_data: non-zero values of CSR matrix
 *   csr_col_index: column indices of non-zero values
 *   csr_row_index: row pointers (size nrows+1)
 *   dense: dense matrix in C-order (row-major), shape (K, ncols)
 *   scale: scalar multiplier
 *   out: output matrix in C-order (row-major), shape (nrows, ncols)
 *   nrows: number of rows in CSR and output
 *   ncols: number of columns in dense and output
 */
template <typename IntT>
void _matmul_csr_dense_c_order(
        const std::complex<double> * _RESTRICT csr_data,
        const IntT * _RESTRICT csr_col_index,
        const IntT * _RESTRICT csr_row_index,
        const std::complex<double> * _RESTRICT dense,
        const std::complex<double> scale,
        std::complex<double> * _RESTRICT out,
        const IntT nrows,
        const IntT ncols);

/**
 * Compute out += scale * (dense @ conj(csr.T)) for F-ordered dense matrices.
 * 
 * For each column of the output (= row of CSR), we iterate over non-zero elements
 * and accumulate: out[:, col] += scale * conj(csr[col, k_idx]) * dense[:, k_idx]
 * 
 * Parameters:
 *   dense: dense matrix in F-order (column-major), shape (nrows, K)
 *   csr_data: non-zero values of CSR matrix
 *   csr_col_index: column indices of non-zero values
 *   csr_row_index: row pointers (size ncols+1)
 *   scale: scalar multiplier
 *   out: output matrix in F-order (column-major), shape (nrows, ncols)
 *   nrows: number of rows in dense and output
 *   ncols: number of columns in output (= rows in CSR)
 */
template <typename IntT>
void _matmul_dag_dense_csr_f_order(
        const std::complex<double> * _RESTRICT dense,
        const std::complex<double> * _RESTRICT csr_data,
        const IntT * _RESTRICT csr_col_index,
        const IntT * _RESTRICT csr_row_index,
        const std::complex<double> scale,
        std::complex<double> * _RESTRICT out,
        const IntT nrows,
        const IntT ncols);

#endif // MATMUL_CSR_DENSE_HPP
