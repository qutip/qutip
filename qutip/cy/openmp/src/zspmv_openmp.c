#include <stddef.h>
#include <complex.h>
#include <omp.h>

#ifdef __SSE3__
#include <pmmintrin.h>

void zspmvpy_openmp(const double complex *restrict data, const int *restrict ind, 
            const int *restrict ptr,
            const double complex *restrict vec, const double complex a, 
            double complex *restrict out, const unsigned int nrows,
            const unsigned int nthr)
{
    size_t row, jj;
    unsigned int row_start, row_end;
    __m128d num1, num2, num3, num4; //Define 2x 64bit float registers
    #pragma omp parallel for \
        private(row,num1,num2,num3,num4,row_start,row_end,jj) \
        shared(data,ind,ptr,vec,out) schedule(static) \
        num_threads(nthr)
    for (row=0; row < nrows; row++)
    {
        num4 = _mm_setzero_pd();
        row_start = ptr[row];
        row_end = ptr[row+1];
        for (jj=row_start; jj <row_end; jj++)
        {
            num1 = _mm_loaddup_pd(&__real__ data[jj]);
            num2 = _mm_set_pd(__imag__ vec[ind[jj]], __real__ vec[ind[jj]]);
            num3 = _mm_mul_pd(num2, num1);
            num1 = _mm_loaddup_pd(&__imag__ data[jj]);
            num2 = _mm_shuffle_pd(num2, num2, 1);
            num2 = _mm_mul_pd(num2, num1);
            num3 = _mm_addsub_pd(num3, num2);
            num4 = _mm_add_pd(num3, num4);
        }
        num1 = _mm_loaddup_pd(&__real__ a);
        num3 = _mm_mul_pd(num4, num1);
        num1 = _mm_loaddup_pd(&__imag__ a);
        num4 = _mm_shuffle_pd(num4, num4, 1);
        num4 = _mm_mul_pd(num4, num1);
        num3 = _mm_addsub_pd(num3, num4);
        num2 = _mm_loadu_pd((double *)&out[row]);
        num3 = _mm_add_pd(num2, num3);
        _mm_storeu_pd((double *)&out[row], num3);
    }
}
#else
void zspmvpy_openmp(const double complex *__restrict__ data, const int *__restrict__ ind, 
            const int *__restrict__ ptr,
            const double complex *__restrict__ vec, const double complex a, 
            double complex *__restrict__ out, const unsigned int nrows, 
            const unsigned int nthr)
{
    size_t row, jj;
    unsigned int row_start, row_end;
    double complex dot;
    #pragma omp parallel for \
        private(row,row_start,row_end,jj,dot) \
        shared(data,ind,ptr,vec,out) schedule(static) \
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