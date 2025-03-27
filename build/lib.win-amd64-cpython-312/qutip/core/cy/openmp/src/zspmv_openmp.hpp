#include <complex>

#ifdef __GNUC__
void zspmvpy_openmp(const std::complex<double> * __restrict__ data, const int * __restrict__ ind,
            const int *__restrict__ ptr,
            const std::complex<double> * __restrict__ vec, const std::complex<double> a,
            std::complex<double> * __restrict__ out,
            const unsigned int nrows, const unsigned int nthr);
#elif defined(_MSC_VER)
void zspmvpy_openmp(const std::complex<double> * __restrict data, const int * __restrict ind,
            const int *__restrict ptr,
            const std::complex<double> * __restrict vec, const std::complex<double> a,
            std::complex<double> * __restrict out,
            const int nrows, const unsigned int nthr);
#else
void zspmvpy_openmp(const std::complex<double> * data, const int * ind,
            const int * ptr,
            const std::complex<double> * vec, const std::complex<double> a,
            std::complex<double> * out,
            const unsigned int nrows, const unsigned int nthr);
#endif
