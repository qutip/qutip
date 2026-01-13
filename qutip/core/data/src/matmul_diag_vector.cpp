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
    const double * scale_dbl = reinterpret_cast<const double *>(&scale);
    double * out_dbl = reinterpret_cast<double *>(out);
    double prod_re, prod_im, scaled_re, scaled_im;
    // Gcc does not vectorize complex automatically?
    for (IntT i=0; i<length*2; i+=2){
        // Compute data[i] * vec[i]
        prod_re = data_dbl[i] * vec_dbl[i] - data_dbl[i+1] * vec_dbl[i+1];
        prod_im = data_dbl[i] * vec_dbl[i+1] + data_dbl[i+1] * vec_dbl[i];
        // Multiply by scale
        scaled_re = scale_dbl[0] * prod_re - scale_dbl[1] * prod_im;
        scaled_im = scale_dbl[0] * prod_im + scale_dbl[1] * prod_re;
        // Accumulate
        out_dbl[i] += scaled_re;
        out_dbl[i+1] += scaled_im;
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
        const IntT width,
        const std::complex<double> scale
){
    const double * data_dbl = reinterpret_cast<const double *>(data);
    const double * vec_dbl = reinterpret_cast<const double *>(vec);
    const double * scale_dbl = reinterpret_cast<const double *>(&scale);
    double * out_dbl = reinterpret_cast<double *>(out);
    IntT ptr = 0;
    double prod_re, prod_im, scaled_re, scaled_im;
    // Gcc does not vectorize complex automatically?
    for (IntT i=0; i<length; i++)
      for (IntT j=0; j<width; j++){
        // Compute data[ptr] * vec[i]
        prod_re = data_dbl[ptr] * vec_dbl[2*i] - data_dbl[ptr+1] * vec_dbl[2*i+1];
        prod_im = data_dbl[ptr] * vec_dbl[2*i+1] + data_dbl[ptr+1] * vec_dbl[2*i];
        // Multiply by scale
        scaled_re = scale_dbl[0] * prod_re - scale_dbl[1] * prod_im;
        scaled_im = scale_dbl[0] * prod_im + scale_dbl[1] * prod_re;
        // Accumulate
        out_dbl[ptr] += scaled_re;
        out_dbl[ptr+1] += scaled_im;
        ptr+=2;
    }
}

template void _matmul_diag_block<>(
        const std::complex<double> * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        std::complex<double> * _RESTRICT,
        const int,
        const int,
        const std::complex<double>);
template void _matmul_diag_block<>(
        const std::complex<double> * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        std::complex<double> * _RESTRICT,
        const long,
        const long,
        const std::complex<double>);
template void _matmul_diag_block<>(
        const std::complex<double> * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        std::complex<double> * _RESTRICT,
        const long long,
        const long long,
        const std::complex<double>);

template <typename IntT>
void _matmul_dag_diag_vector(
        const std::complex<double> * _RESTRICT data,
        const std::complex<double> * _RESTRICT vec,
        std::complex<double> * _RESTRICT out,
        const IntT length,
        const std::complex<double> scale
){
    const double * data_dbl = reinterpret_cast<const double *>(data);
    const double * vec_dbl = reinterpret_cast<const double *>(vec);
    const double * scale_dbl = reinterpret_cast<const double *>(&scale);
    double * out_dbl = reinterpret_cast<double *>(out);
    double prod_re, prod_im, scaled_re, scaled_im;
    // Conjugate the diagonal data: conj(data[i]) * vec[i]
    // conj(a + bi) = a - bi
    // (a - bi) * (c + di) = ac + bd + i(ad - bc)
    for (IntT i=0; i<length*2; i+=2){
        // Compute conj(data[i]) * vec[i]
        prod_re = data_dbl[i] * vec_dbl[i] + data_dbl[i+1] * vec_dbl[i+1];
        prod_im = data_dbl[i] * vec_dbl[i+1] - data_dbl[i+1] * vec_dbl[i];
        // Multiply by scale
        scaled_re = scale_dbl[0] * prod_re - scale_dbl[1] * prod_im;
        scaled_im = scale_dbl[0] * prod_im + scale_dbl[1] * prod_re;
        // Accumulate
        out_dbl[i] += scaled_re;
        out_dbl[i+1] += scaled_im;
    }
}

template void _matmul_dag_diag_vector<>(
        const std::complex<double> * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        std::complex<double> * _RESTRICT,
        const int,
        const std::complex<double>);
template void _matmul_dag_diag_vector<>(
        const std::complex<double> * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        std::complex<double> * _RESTRICT,
        const long,
        const std::complex<double>);
template void _matmul_dag_diag_vector<>(
        const std::complex<double> * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        std::complex<double> * _RESTRICT,
        const long long,
        const std::complex<double>);

template <typename IntT>
void _matmul_dag_diag_block(
        const std::complex<double> * _RESTRICT data,
        const std::complex<double> * _RESTRICT vec,
        std::complex<double> * _RESTRICT out,
        const IntT length,
        const IntT width,
        const std::complex<double> scale
){
    const double * data_dbl = reinterpret_cast<const double *>(data);
    const double * vec_dbl = reinterpret_cast<const double *>(vec);
    const double * scale_dbl = reinterpret_cast<const double *>(&scale);
    double * out_dbl = reinterpret_cast<double *>(out);
    IntT ptr = 0;
    double prod_re, prod_im, scaled_re, scaled_im;
    // Conjugate the diagonal data: conj(data[i]) * vec[j]
    // conj(a + bi) = a - bi
    // (a - bi) * (c + di) = ac + bd + i(ad - bc)
    for (IntT i=0; i<length; i++)
      for (IntT j=0; j<width; j++){
        // Compute conj(data[i]) * vec[ptr]
        prod_re = data_dbl[2*i] * vec_dbl[ptr] + data_dbl[2*i+1] * vec_dbl[ptr+1];
        prod_im = data_dbl[2*i] * vec_dbl[ptr+1] - data_dbl[2*i+1] * vec_dbl[ptr];
        // Multiply by scale
        scaled_re = scale_dbl[0] * prod_re - scale_dbl[1] * prod_im;
        scaled_im = scale_dbl[0] * prod_im + scale_dbl[1] * prod_re;
        // Accumulate
        out_dbl[ptr] += scaled_re;
        out_dbl[ptr+1] += scaled_im;
        ptr+=2;
    }
}

template void _matmul_dag_diag_block<>(
        const std::complex<double> * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        std::complex<double> * _RESTRICT,
        const int,
        const int,
        const std::complex<double>);
template void _matmul_dag_diag_block<>(
        const std::complex<double> * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        std::complex<double> * _RESTRICT,
        const long,
        const long,
        const std::complex<double>);
template void _matmul_dag_diag_block<>(
        const std::complex<double> * _RESTRICT,
        const std::complex<double> * _RESTRICT,
        std::complex<double> * _RESTRICT,
        const long long,
        const long long,
        const std::complex<double>);
