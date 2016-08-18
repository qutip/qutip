#include <stddef.h>
#include <math.h>

inline double phi(double t)
{
    double abs_t = fabs(t);
    if ( abs_t <= 1)
    {
        return 4 - 6 * pow(abs_t, 2) + 3 * pow(abs_t, 3);
    }
    else if ( abs_t <= 2)
    {
        return pow(2-abs_t, 3);
    }
    else
    {
        return 0;
    }
}

double cinterpolate(double x, double a, double b, double *c, int lenc)
{
    int n = lenc - 3;
    double h = (b-a) / n;
    int l = (int)((x-a)/h) + 1;
    int m = (int)(fmin(l+3, n+3));
    size_t ii;
    double s = 0;
    double pos = (x-a)/h + 2;
    
    for (ii = l; ii < m+1; ii++)
    {
        s += c[ii-1] * phi(pos - ii);
    }
    return s;    
}

void carray_interpolate(double * x, double a, double b, double *c, double *out, int lenx, int lenc)
{
    int n = lenc - 3;
    double h = (b-a) / n;
    size_t ii, jj;
    int l, m;
    double pos;
    
    for (jj=0; jj < lenx; jj++)
    {
        l = (int)((x[jj]-a)/h) + 1;
        m = (int)(fmin(l+3, n+3));
        pos = (x[jj]-a)/h + 2;
        for (ii = l; ii < m+1; ii++)
        {
            out[jj] += c[ii-1] * phi(pos - ii);
        }
    }
}