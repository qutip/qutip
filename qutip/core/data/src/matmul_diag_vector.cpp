#include <complex>

#include "matmul_diag_vector.hpp"

template <typename IntT>
void _matmul_diag_vector(
        const std::complex<double> * _RESTRICT data,
        const std::complex<double> * _RESTRICT vec,
        std::complex<double> * _RESTRICT out,
        const IntT length,
        const std::complex<double> scale
){
    const double * data_dbl = reinterpret_cast<const double *>(data);
    const double * vec_dbl = reinterpret_cast<const double *>(vec);
    double * out_dbl = reinterpret_cast<double *>(out);
    // Gcc does not vectorize complex automatically?
    for (IntT i=0; i<length*2; i+=2){
        out_dbl[i] += data_dbl[i] * vec_dbl[i];
        out_dbl[i] -= data_dbl[i+1] * vec_dbl[i+1];
        out_dbl[i+1] += data_dbl[i] * vec_dbl[i+1];
        out_dbl[i+1] += data_dbl[i+1] * vec_dbl[i];
    }
}

template void _matmul_diag_vector<>(
        const std::complex<double> * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        std::complex<double> * _RESTRICT,
        const int,
        const std::complex<double>);
template void _matmul_diag_vector<>(
        const std::complex<double> * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        std::complex<double> * _RESTRICT,
        const long,
        const std::complex<double>);
template void _matmul_diag_vector<>(
        const std::complex<double> * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        std::complex<double> * _RESTRICT,
        const long long,
        const std::complex<double>);

template <typename IntT>
void _matmul_diag_block(
        const std::complex<double> * _RESTRICT data,
        const std::complex<double> * _RESTRICT vec,
        std::complex<double> * _RESTRICT out,
        const IntT length,
        const IntT width
){
    const double * data_dbl = reinterpret_cast<const double *>(data);
    const double * vec_dbl = reinterpret_cast<const double *>(vec);
    double * out_dbl = reinterpret_cast<double *>(out);
    IntT ptr = 0;
    // Gcc does not vectorize complex automatically?
    for (IntT i=0; i<length; i++)
      for (IntT j=0; j<width; j++){
        out_dbl[ptr] += data_dbl[ptr] * vec_dbl[2*i];
        out_dbl[ptr] -= data_dbl[ptr+1] * vec_dbl[2*i+1];
        out_dbl[ptr+1] += data_dbl[ptr] * vec_dbl[2*i+1];
        out_dbl[ptr+1] += data_dbl[ptr+1] * vec_dbl[2*i];
        ptr+=2;
    }
}

template void _matmul_diag_block<>(
        const std::complex<double> * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        std::complex<double> * _RESTRICT,
        const int,
        const int);
template void _matmul_diag_block<>(
        const std::complex<double> * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        std::complex<double> * _RESTRICT,
        const long,
        const long);
template void _matmul_diag_block<>(
        const std::complex<double> * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        std::complex<double> * _RESTRICT,
        const long long,
        const long long);
