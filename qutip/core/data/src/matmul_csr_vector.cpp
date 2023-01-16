#include <complex>
#if \
    (defined(__GNUC__) && defined(__SSE3__))\
    || (defined(_MSC_VER) && defined(__AVX__))
/* If we're going to manually do the vectorisation, we need to make sure we've
 * included the preprocessor directives.
 */
# include <pmmintrin.h>
#endif

#include "matmul_csr_vector.hpp"

template <typename IntT>
#if \
    (defined(__GNUC__) && defined(__SSE3__))\
    || (defined(_MSC_VER) && defined(__AVX__))
/* Manually apply the vectorisation. */
void _matmul_csr_vector(
        const std::complex<double> * _RESTRICT data,
        const IntT * _RESTRICT col_index,
        const IntT * _RESTRICT row_index,
        const std::complex<double> * _RESTRICT vec,
        const std::complex<double> scale,
        std::complex<double> * _RESTRICT out,
        const IntT nrows)
{
    IntT row_start, row_end;
    __m128d num1, num2, num3, num4;
    for (IntT row=0; row < nrows; row++) {
        num4 = _mm_setzero_pd();
        row_start = row_index[row];
        row_end = row_index[row+1];
        for (IntT ptr=row_start; ptr < row_end; ptr++) {
            num1 = _mm_loaddup_pd(&reinterpret_cast<const double(&)[2]>(data[ptr])[0]);
            num2 = _mm_set_pd(std::imag(vec[col_index[ptr]]),
                              std::real(vec[col_index[ptr]]));
            num3 = _mm_mul_pd(num2, num1);
            num1 = _mm_loaddup_pd(&reinterpret_cast<const double(&)[2]>(data[ptr])[1]);
            num2 = _mm_shuffle_pd(num2, num2, 1);
            num2 = _mm_mul_pd(num2, num1);
            num3 = _mm_addsub_pd(num3, num2);
            num4 = _mm_add_pd(num3, num4);
        }
        num1 = _mm_loaddup_pd(&reinterpret_cast<const double(&)[2]>(scale)[0]);
        num3 = _mm_mul_pd(num4, num1);
        num1 = _mm_loaddup_pd(&reinterpret_cast<const double(&)[2]>(scale)[1]);
        num4 = _mm_shuffle_pd(num4, num4, 1);
        num4 = _mm_mul_pd(num4, num1);
        num3 = _mm_addsub_pd(num3, num4);
        num2 = _mm_loadu_pd((double *)&out[row]);
        num3 = _mm_add_pd(num2, num3);
        _mm_storeu_pd((double *)&out[row], num3);
    }
}
#else
/* No manual vectorisation. */
void _matmul_csr_vector(
        const std::complex<double> * _RESTRICT data,
        const IntT * _RESTRICT col_index,
        const IntT * _RESTRICT row_index,
        const std::complex<double> * _RESTRICT vec,
        const std::complex<double> scale,
        std::complex<double> * _RESTRICT out,
        const IntT nrows)
{
    IntT row_start, row_end;
    std::complex<double> dot;
    for (size_t row=0; row < nrows; row++)
    {
        dot = 0;
        row_start = row_index[row];
        row_end = row_index[row+1];
        for (size_t ptr=row_start; ptr < row_end; ptr++)
        {
            dot += data[ptr]*vec[col_index[ptr]];
        }
        out[row] += scale * dot;
    }
}
#endif

/* It seems wrong to me to specify the integer specialisations as `int`, `long` and
 * `long long` rather than just `int32_t` and `int64_t`, but for some reason the
 * latter causes compatibility issues with defining the sized types with the
 * numpy `cnp.npy_int32` and `cnp.npy_int64` typedefs.  We need to specify all
 * three of `int`, `long` and `long long` when doing things this way (despite
 * the almost certain duplication) because on Unix-likes (where `int`[`long`] is
 * typically 32[64]-bit) numpy typedef's int32 to int and int64 to long, whereas
 * on Windows (where `int` and `long` are both 32-bit), it typedef's to `long`
 * and `long long`.
 * - Jake Lishman 2020-08-10.
 */
template void _matmul_csr_vector<>(
        const std::complex<double> * _RESTRICT,
        const int * _RESTRICT,
        const int * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        const std::complex<double>,
        std::complex<double> * _RESTRICT,
        const int);
template void _matmul_csr_vector<>(
        const std::complex<double> * _RESTRICT,
        const long * _RESTRICT,
        const long * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        const std::complex<double>,
        std::complex<double> * _RESTRICT,
        const long);
template void _matmul_csr_vector<>(
        const std::complex<double> * _RESTRICT,
        const long long * _RESTRICT,
        const long long * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        const std::complex<double>,
        std::complex<double> * _RESTRICT,
        const long long);
