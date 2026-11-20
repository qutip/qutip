#include <complex>

#if defined(__GNUC__) || defined(_MSC_VER)
# define _RESTRICT __restrict
#else
# define _RESTRICT
#endif

template <typename IntT>
void _matmul_csr_vector(
        const std::complex<double> * _RESTRICT data,
        const IntT * _RESTRICT col_index,
        const IntT * _RESTRICT row_index,
        const std::complex<double> * _RESTRICT vec,
        const std::complex<double> scale,
        std::complex<double> * _RESTRICT out,
        const IntT nrows);

template <typename IntT>
void _matmul_dag_csr_vector(
        const std::complex<double> * _RESTRICT data,
        const IntT * _RESTRICT col_index,
        const IntT * _RESTRICT row_index,
        const std::complex<double> * _RESTRICT vec,
        const std::complex<double> scale,
        std::complex<double> * _RESTRICT out,
        const IntT nrows);
