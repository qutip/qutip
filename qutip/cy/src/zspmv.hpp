#include <complex>

#ifdef __GNUC__
void zspmvpy(const std::complex<double> * __restrict__ data, const int * __restrict__ ind, 
            const int *__restrict__ ptr,
            const std::complex<double> * __restrict__ vec, const std::complex<double> a, 
            std::complex<double> * __restrict__ out,
            const unsigned int nrows);
#else
void zspmvpy(const std::complex<double> * __restrict data, const int * __restrict ind, 
            const int *__restrict ptr,
            const std::complex<double> * __restrict vec, const std::complex<double> a, 
            std::complex<double> * __restrict out,
            const unsigned int nrows);
#endif