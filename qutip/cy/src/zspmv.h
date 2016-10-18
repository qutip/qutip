#include <complex.h>

void zspmvpy(double complex *data, int *ind, int *ptr,
            double complex *vec, double complex a, double complex *out, 
            int nrows);
