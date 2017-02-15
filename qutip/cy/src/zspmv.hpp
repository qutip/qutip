#include <complex>

void zspmvpy(const std::complex<double> * __restrict__ data, const int * __restrict__ ind, 
            const int *__restrict__ ptr,
            const std::complex<double> * __restrict__ vec, const std::complex<double> a, 
            std::complex<double> * __restrict__ out,
            const unsigned int nrows);
