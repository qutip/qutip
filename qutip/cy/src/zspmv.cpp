#include <complex>

#ifdef __SSE3__
#include <pmmintrin.h>
void zspmvpy(const std::complex<double> * __restrict__ data, const int * __restrict__ ind,
            const int * __restrict__ ptr,
            const std::complex<double> * __restrict__ vec, const std::complex<double> a,
            std::complex<double> * __restrict__ out, const unsigned int nrows)
{
    size_t row, jj;
    unsigned int row_start, row_end;
    __m128d num1, num2, num3, num4; //Define 2x 64bit float registers
    for (row=0; row < nrows; row++)
    {
        num4 = _mm_setzero_pd();
        row_start = ptr[row];
        row_end = ptr[row+1];
        for (jj=row_start; jj <row_end; jj++)
        {
            num1 = _mm_loaddup_pd(&reinterpret_cast<const double(&)[2]>(data[jj])[0]);
            num2 = _mm_set_pd(std::imag(vec[ind[jj]]),std::real(vec[ind[jj]]));
            num3 = _mm_mul_pd(num2, num1);
            num1 = _mm_loaddup_pd(&reinterpret_cast<const double(&)[2]>(data[jj])[1]);
            num2 = _mm_shuffle_pd(num2, num2, 1);
            num2 = _mm_mul_pd(num2, num1);
            num3 = _mm_addsub_pd(num3, num2);
            num4 = _mm_add_pd(num3, num4);
        }
        num1 = _mm_loaddup_pd(&reinterpret_cast<const double(&)[2]>(a)[0]);
        num3 = _mm_mul_pd(num4, num1);
        num1 = _mm_loaddup_pd(&reinterpret_cast<const double(&)[2]>(a)[1]);
        num4 = _mm_shuffle_pd(num4, num4, 1);
        num4 = _mm_mul_pd(num4, num1);
        num3 = _mm_addsub_pd(num3, num4);
        num2 = _mm_loadu_pd((double *)&out[row]);
        num3 = _mm_add_pd(num2, num3);
        _mm_storeu_pd((double *)&out[row], num3);
    }
}
#else
void zspmvpy(const std::complex<double> * __restrict__ data, const int * __restrict__ ind,
            const int * __restrict__ ptr,
            const std::complex<double> * __restrict__ vec, const std::complex<double> a,
            std::complex<double> * __restrict__ out, const unsigned int nrows)
{
    size_t row, jj;
    unsigned int row_start, row_end;
    double complex dot;
    for (row=0; row < nrows; row++)
    {
        dot = 0;
        row_start = ptr[row];
        row_end = ptr[row+1];
        for (jj=row_start; jj <row_end; jj++)
        {
            dot += data[jj]*vec[ind[jj]];
        }
        out[row] += a*dot;
    }
}
#endif