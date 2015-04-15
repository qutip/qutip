# -*- coding: utf-8 -*-
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2014 and later, Alexander J G Pitchford, Jonathan Zoller
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
Classes and support functions for producing pulses for the CRAB algorithm
"""

import numpy as np
import qutip.control.pulsegen as pulsegen

# AJGP 2015-05-14: 
# The intention is to have a more general base class that allows
# setting of general basis functions
class PulseGenCrabFourier(pulsegen.PulseGen):
    """
    Generates a pulse using the Fourier basis functions, i.e. sin and cos

    Attributes
    ----------
    num_coeffs : integer
        Number of coefficients used for each basis function
        
    num_basis_funcs : integer
        Number of basis functions
        In this case set at 2 and should not be changed
        
    coeffs : float array[num_coeffs, num_basis_funcs]
        The basis coefficient values

    freqs : float array[num_coeffs]
        Frequencies for the basis functions
    """

    def reset(self):
        """
        reset attributes to default values
        """
        PulseGen.reset(self)

        self.num_coeffs = 10
        self.num_basis_funcs = 2
        self.coeffs = None
        self.freqs = None
        self.time = None

    def init_pulse(self, num_coeffs=None):
        """
        Set the initial freq and coefficient values
        """
        pulsegen.PulseGen.init_pulse(self)
        self.time = np.zeros(self.num_tslots, dtype=float)
        for k in range(self.num_tslots-1):
            self.time[k+1] = self.time[k] + self.tau[k]
            
        self.init_coeffs(num_coeffs=num_coeffs)
        self.init_freqs()
        
    def init_freqs(self):
        """
        Generate the frequencies
        These are the Fourier harmonics with a uniformly distributed
        random offset
        """
        self.freqs = np.empty(self.num_coeffs)
        ff = 2*np.pi / self.pulse_time
        for i in range(self.num_coeffs):
            self.freqs[i] = ff*(i + 1)
            
        self.freqs += np.random.random(self.num_coeffs) - 0.5
        
    def init_coeffs(self, num_coeffs=None):
        """
        Generate the initial ceofficent values.
        
        Parameters
        ----------
        num_coeffs : integer
            Number of coefficients used for each basis function
            If given this overides the default and sets the attribute
            of the same name.
        """
        if num_coeffs:
            self.num_coeffs = num_coeffs
        # For now just use the scaling and offset attributes
        r = np.random.random([self.num_coeffs, self.num_basis_funcs])
        self.coeffs = (2*r - 1) * self.scaling + self.offset
        

    def gen_pulse(self, coeffs=None):
        """
        Generate a pulse using the Fourier basis with the freqs and
        coeffs attributes.
        
        Parameters
        ----------
        coeffs : float array[num_coeffs, num_basis_funcs]
            The basis coefficient values
            If given this overides the default and sets the attribute
            of the same name.
        """
        if coeffs:
            self.coeffs = coeffs
            
        if not self._pulse_initialised:
            self.init_pulse()
        
        pulse = np.zeros(self.num_tslots)

        for i in range(self.num_coeffs):
            pulse += coeffs[i, 0]*np.sin(self.freqs[i]*self.time) + \
                        coeffs[i, 1]*np.cos(self.freqs[i]*self.time) 

        return pulse
    