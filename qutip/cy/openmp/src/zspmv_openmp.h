#include <complex.h>

void zspmvpy_openmp(double complex *data, int *ind, int *ptr,
            double complex *vec, double complex a, double complex *out, 
            int nrows, int nthr);
