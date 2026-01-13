#include <complex>
#include "matmul_csr_dense.hpp"

/**
 * CSR @ Dense multiplication for C-ordered (row-major) dense matrices.
 * 
 * Computes: out += scale * (csr @ dense)
 * 
 * Loop structure:
 *   for row in 0..nrows:           // sparse row = output row
 *     for ptr in row_index[row]..row_index[row+1]:
 *       val = scale * csr_data[ptr]
 *       col_idx = csr_col_index[ptr]  // sparse col = dense row
 *       for col in 0..ncols:          // dense col = output col
 *         out[row, col] += val * dense[col_idx, col]
 * 
 * The inner loop operates on contiguous memory for both out[row,:] and 
 * dense[col_idx,:] since both are C-ordered (row-major).
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
        const IntT ncols)
{
    IntT row_start, row_end, col_idx;
    std::complex<double> scaled_val;
    const std::complex<double> *dense_row;
    std::complex<double> *out_row;
    
    // Extract real and imaginary parts for better vectorization
    double val_re, val_im;
    double dense_re, dense_im;
    double *out_ptr;
    const double *dense_ptr;
    
    for (IntT row = 0; row < nrows; row++) {
        row_start = csr_row_index[row];
        row_end = csr_row_index[row + 1];
        out_row = out + row * ncols;
        
        for (IntT ptr = row_start; ptr < row_end; ptr++) {
            scaled_val = scale * csr_data[ptr];
            val_re = std::real(scaled_val);
            val_im = std::imag(scaled_val);
            col_idx = csr_col_index[ptr];
            dense_row = dense + col_idx * ncols;
            
            // Cast to double arrays for explicit real/imag access
            // std::complex<double> is guaranteed to be layout-compatible with double[2]
            out_ptr = reinterpret_cast<double*>(out_row);
            dense_ptr = reinterpret_cast<const double*>(dense_row);
            
            // Inner loop: accumulate val * dense_row[:] into out_row[:]
            // Separate real and imaginary parts to help vectorization
            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            for (IntT col = 0; col < ncols; col++) {
                dense_re = dense_ptr[2*col];
                dense_im = dense_ptr[2*col + 1];
                out_ptr[2*col] += val_re * dense_re - val_im * dense_im;
                out_ptr[2*col + 1] += val_re * dense_im + val_im * dense_re;
            }
        }
    }
}

/**
 * Dense @ CSR† multiplication for F-ordered (column-major) dense matrices.
 * 
 * Computes: out += scale * (dense @ conj(csr.T))
 * 
 * Where:
 *   dense is F-ordered (nrows × K)
 *   csr is (ncols × K), so csr† is (K × ncols)
 *   out is F-ordered (nrows × ncols)
 * 
 * Loop structure:
 *   for col in 0..ncols:           // output column = csr row
 *     for ptr in row_index[col]..row_index[col+1]:
 *       val = scale * conj(csr_data[ptr])
 *       k_idx = csr_col_index[ptr]  // csr col = dense col
 *       for row in 0..nrows:        // dense row = output row
 *         out[row, col] += val * dense[row, k_idx]
 * 
 * The inner loop operates on contiguous memory for both out[:,col] and 
 * dense[:,k_idx] since both are F-ordered (column-major).
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
        const IntT ncols)
{
    IntT row_start, row_end, k_idx;
    std::complex<double> scaled_val;
    const std::complex<double> *dense_col;
    std::complex<double> *out_col;
    
    // Extract real and imaginary parts for better vectorization
    double val_re, val_im;
    double dense_re, dense_im;
    double *out_ptr;
    const double *dense_ptr;
    
    for (IntT col = 0; col < ncols; col++) {
        row_start = csr_row_index[col];
        row_end = csr_row_index[col + 1];
        out_col = out + col * nrows;
        
        for (IntT ptr = row_start; ptr < row_end; ptr++) {
            // Conjugate the CSR data for the adjoint
            scaled_val = scale * std::conj(csr_data[ptr]);
            val_re = std::real(scaled_val);
            val_im = std::imag(scaled_val);
            k_idx = csr_col_index[ptr];
            dense_col = dense + k_idx * nrows;
            
            // Cast to double arrays for explicit real/imag access
            out_ptr = reinterpret_cast<double*>(out_col);
            dense_ptr = reinterpret_cast<const double*>(dense_col);
            
            // Inner loop: accumulate val * dense_col[:] into out_col[:]
            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            for (IntT row = 0; row < nrows; row++) {
                dense_re = dense_ptr[2*row];
                dense_im = dense_ptr[2*row + 1];
                out_ptr[2*row] += val_re * dense_re - val_im * dense_im;
                out_ptr[2*row + 1] += val_re * dense_im + val_im * dense_re;
            }
        }
    }
}

// Explicit template instantiations for common integer types
template void _matmul_csr_dense_c_order<>(
        const std::complex<double> * _RESTRICT,
        const int * _RESTRICT,
        const int * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        const std::complex<double>,
        std::complex<double> * _RESTRICT,
        const int,
        const int);

template void _matmul_csr_dense_c_order<>(
        const std::complex<double> * _RESTRICT,
        const long * _RESTRICT,
        const long * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        const std::complex<double>,
        std::complex<double> * _RESTRICT,
        const long,
        const long);

template void _matmul_csr_dense_c_order<>(
        const std::complex<double> * _RESTRICT,
        const long long * _RESTRICT,
        const long long * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        const std::complex<double>,
        std::complex<double> * _RESTRICT,
        const long long,
        const long long);


template void _matmul_dag_dense_csr_f_order<>(
        const std::complex<double> * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        const int * _RESTRICT,
        const int * _RESTRICT,
        const std::complex<double>,
        std::complex<double> * _RESTRICT,
        const int,
        const int);

template void _matmul_dag_dense_csr_f_order<>(
        const std::complex<double> * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        const long * _RESTRICT,
        const long * _RESTRICT,
        const std::complex<double>,
        std::complex<double> * _RESTRICT,
        const long,
        const long);

template void _matmul_dag_dense_csr_f_order<>(
        const std::complex<double> * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        const long long * _RESTRICT,
        const long long * _RESTRICT,
        const std::complex<double>,
        std::complex<double> * _RESTRICT,
        const long long,
        const long long);
