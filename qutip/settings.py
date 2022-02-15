# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
"""
This module contains settings for the QuTiP graphics, multiprocessing, and
tidyup functionality, etc.
"""
# from __future__ import absolute_import
# import qutip.configrc as qrc
import os, sys
from .utilities import _blas_info
from ctypes import cdll
import platform

__all__ = ['settings']

def _find_mkl():
    """
    Finds the MKL runtime library for the Anaconda and Intel Python
    distributions.
    """
    has_mkl = False
    mkl_lib = None
    if _blas_info() == 'INTEL MKL':
        plat = sys.platform
        python_dir = os.path.dirname(sys.executable)
        if plat in ['darwin','linux2', 'linux']:
            python_dir = os.path.dirname(python_dir)

        if plat == 'darwin':
            lib = '/libmkl_rt.dylib'
        elif plat == 'win32':
            lib = r'\mkl_rt.dll'
        elif plat in ['linux2', 'linux']:
            lib = '/libmkl_rt.so'
        else:
            raise Exception('Unknown platfrom.')

        if plat in ['darwin','linux2', 'linux']:
            lib_dir = '/lib'
        else:
            lib_dir = r'\Library\bin'
        # Try in default Anaconda location first
        try:
            mkl_lib = cdll.LoadLibrary(python_dir+lib_dir+lib)
            has_mkl = True
        except:
            pass

        # Look in Intel Python distro location
        if not has_mkl:
            if plat in ['darwin','linux2', 'linux']:
                lib_dir = '/ext/lib'
            else:
                lib_dir = r'\ext\lib'
            try:
                mkl_lib = \
                    cdll.LoadLibrary(python_dir+lib_dir+lib)
                has_mkl = True
            except:
                pass
    return has_mkl, mkl_lib


class Settings:
    """
    Qutip default settings and options.
    `print(qutip.settings)` to list all available options.
    """
    def __init__(self):
        self._has_mkl, self._mkl_lib = _find_mkl()
        try:
            self.tmproot = os.path.join(os.path.expanduser("~"), '.qutip')
        except OSError:
            self._tmproot = "."
        self.core = None
        self.compilation = None
        self._solvers = []
        self._integrators = []

    @property
    def has_mkl(self):
        return self._has_mkl

    @property
    def mkl_lib(self):
        return self._mkl_lib

    @property
    def ipython(self):
        try:
            __IPYTHON__
            return True
        except NameError:
            return False

    @property
    def eigh_unsafe(self):
        return _blas_info() == "OPENBLAS" and platform.system() == 'Darwin'

    @property
    def tmproot(self):
        return self._tmproot

    @tmproot.setter
    def tmproot(self, root):
        if not os.path.exists(root):
            os.mkdir(root)
        self._tmproot = root

    @property
    def coeffroot(self):
        return self._coeffroot

    @coeffroot.setter
    def coeffroot(self, root):
        if not os.path.exists(root):
            os.mkdir(root)
        if root not in sys.path:
            sys.path.insert(0, root)
        self._coeffroot = root

    @property
    def coeff_write_ok(self):
        return os.access(self.coeffroot, os.W_OK)

    @property
    def has_openmp(self):
        return False
        # We keep this as a reminder for when openmp is restored: see Pull #652
        # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


settings = Settings()
