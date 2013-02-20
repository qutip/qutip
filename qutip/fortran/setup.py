#!/usr/bin/env python
from os.path import join


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, NotFoundError

    config = Configuration('fortran', parent_package, top_path)

    sources = [
        'qutraj_run.pyf',
        'qutraj_precision.f90',
        'mt19937.f90',
        'linked_list.f90',
        'qutraj_general.f90',
        'qutraj_hilbert.f90',
        'qutraj_evolve.f90',
    ]

    libs = ['zvode']

    config.add_library('zvode', sources=[join('zvode', '*.f')])

    #
    # LAPACK?
    #
    lapack_opt = get_info('lapack', notfound_action=1)

    if not lapack_opt:
        # raise NotFoundError,'no lapack resources found'
        print("Warning: No lapack resource found. Linear algebra routines"
              + " like 'eigenvalues' and 'entropy' will not be available.")
        sources.append('qutraj_nolinalg.f90')
    else:
        sources.append('qutraj_linalg.f90')
        libs.extend(lapack_opt['libraries'])

    #
    # BLAS
    #
    if not lapack_opt:
        blas_opt = get_info('blas', notfound_action=2)
    else:
        blas_opt = lapack_opt

    # Remove libraries key from blas_opt
    if 'libraries' in blas_opt:  # key doesn't exist on OS X ...
        libs.extend(blas_opt['libraries'])
    newblas = {}
    for key in blas_opt.keys():
        if key == 'libraries':
            continue
        newblas[key] = blas_opt[key]

    # Add this last
    sources.append('qutraj_run.f90')

    config.add_extension('qutraj_run',
                         sources=sources,
                         extra_compile_args=[
                             #'-DF2PY_REPORT_ON_ARRAY_COPY=1',
                         ],
                         libraries=libs,
                         **newblas
                         )

    return config


if (__name__ == '__main__'):
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
