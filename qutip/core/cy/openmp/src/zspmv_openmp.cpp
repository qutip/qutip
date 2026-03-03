#include <complex>
#include <omp.h>

#if defined(__GNUC__) && defined(__SSE3__) // Using GCC or CLANG and SSE3
#include <pmmintrin.h>
void zspmvpy_openmp(const std::complex<double> * __restrict__ data, const int * __restrict__ ind,
            const int * __restrict__ ptr,
            const std::complex<double> * __restrict__ vec, const std::complex<double> a,
            std::complex<double> * __restrict__ out, const unsigned int nrows,
            const unsigned int nthr)
{
    size_t row, jj;
    unsigned int row_start, row_end;
    __m128d num1, num2, num3, num4;
    #pragma omp parallel for \
        private(row,num1,num2,num3,num4,row_start,row_end,jj) \
        shared(data,ind,ptr,out,vec) schedule(static) \
        num_threads(nthr)
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
#elif defined(__GNUC__) // Using GCC or CLANG but no SSE3
void zspmvpy_openmp(const std::complex<double> * __restrict__ data, const int * __restrict__ ind,
            const int * __restrict__ ptr,
            const std::complex<double> * __restrict__ vec, const std::complex<double> a,
            std::complex<double> * __restrict__ out, const unsigned int nrows,
            const unsigned int nthr)
{
    size_t row, jj;
    unsigned int row_start, row_end;
    std::complex<double> dot;
    #pragma omp parallel for \
        private(row,dot,row_start,row_end,jj) \
        shared(data,ind,ptr,out,vec) schedule(static) \
        num_threads(nthr)
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
#elif defined(_MSC_VER) && defined(__AVX__) // Visual Studio with AVX
#include <pmmintrin.h>
void zspmvpy_openmp(const std::complex<double> * __restrict data, const int * __restrict ind,
            const int * __restrict ptr,
            const std::complex<double> * __restrict vec, const std::complex<double> a,
            std::complex<double> * __restrict out, const int nrows,
            const unsigned int nthr)
{
    int row, jj;
    int row_start, row_end;
    __m128d num1, num2, num3, num4;
    #pragma omp parallel for \
        private(row,num1,num2,num3,num4,row_start,row_end,jj) \
        shared(data,ind,ptr,out,vec) schedule(static) \
        num_threads(nthr)
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
#elif defined(_MSC_VER) // Visual Studio no AVX
void zspmvpy_openmp(const std::complex<double> * __restrict data, const int * __restrict ind,
            const int * __restrict ptr,
            const std::complex<double> * __restrict vec, const std::complex<double> a,
            std::complex<double> * __restrict out, const int nrows,
            const unsigned int nthr)
{
    int row, jj;
    int row_start, row_end;
    std::complex<double> dot;
    #pragma omp parallel for \
        private(row,dot,row_start,row_end,jj) \
        shared(data,ind,ptr,out,vec) schedule(static) \
        num_threads(nthr)
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
#else // Everything else
void zspmvpy_openmp(const std::complex<double> * data, const int * ind,
            const int * ptr,
            const std::complex<double> * vec, const std::complex<double> a,
            std::complex<double> * out, const unsigned int nrows,
            const unsigned int nthr)
{
    size_t row, jj;
    unsigned int row_start, row_end;
    std::complex<double> dot;
    #pragma omp parallel for \
        private(row,dot,row_start,row_end,jj) \
        shared(data,ind,ptr,out,vec) schedule(static) \
        num_threads(nthr)
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
