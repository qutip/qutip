# -*- coding: utf-8 -*-
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2014 and later, Alexander J G Pitchford
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

# @author: Alexander Pitchford
# @email1: agp1@aber.ac.uk
# @email2: alex.pitchford@gmail.com
# @organization: Aberystwyth University
# @supervisor: Daniel Burgarth

"""
Pulse generator - Generate pulses for the timeslots
Each class defines a gen_pulse function that produces a float array of
size num_tslots. Each class produces a differ type of pulse.
See the class and gen_pulse function descriptions for details
"""

import numpy as np
import qutip.control.dynamics as dynamics
import qutip.control.errors as errors


def create_pulse_gen(pulse_type='RND', dyn=None):
    """
    Create and return a pulse generator object matching the given type.
    The pulse generators each produce a different type of pulse,
    see the gen_pulse function description for details.
    These are the random pulse options:
        RND - Independent random value in each timeslot
        RNDFOURIER - Fourier series with random coefficients
        RNDWAVES - Summation of random waves
        RNDWALK1 - Random change in amplitude each timeslot
        RNDWALK2 - Random change in amp gradient each timeslot
    These are the other non-periodic options:
        LIN - Linear, i.e. contant gradient over the time
        ZERO - special case of the LIN pulse, where the gradient is 0
    These are the periodic options
        SINE - Sine wave
        SQUARE - Square wave
        SAW - Saw tooth wave
        TRIANGLE - Triangular wave
    If a Dynamics object is passed in then this is used in instantiate
    the PulseGen, meaning that some timeslot and amplitude properties
    are copied over.
    """

    if pulse_type == 'RND':
        return PulseGenRandom(dyn)
    if pulse_type == 'RNDFOURIER':
        return PulseGenRndFourier(dyn)
    if pulse_type == 'RNDWAVES':
        return PulseGenRndWaves(dyn)
    if pulse_type == 'RNDWALK1':
        return PulseGenRndWalk1(dyn)
    if pulse_type == 'RNDWALK2':
        return PulseGenRndWalk2(dyn)
    elif pulse_type == 'LIN':
        return PulseGenLinear(dyn)
    elif pulse_type == 'ZERO':
        return PulseGenZero(dyn)
    elif pulse_type == 'SINE':
        return PulseGenSine(dyn)
    elif pulse_type == 'SQUARE':
        return PulseGenSquare(dyn)
    elif pulse_type == 'SAW':
        return PulseGenSaw(dyn)
    elif pulse_type == 'TRIANGLE':
        return PulseGenTriangle(dyn)
    else:
        raise ValueError("No option for pulse_type '{}'".format(pulse_type))


class PulseGen:
    """
    Pulse generator
    Base class for all Pulse generators
    The object can optionally be instantiated with a Dynamics object,
    in which case the timeslots and amplitude scaling and offset
    are copied from that.
    Otherwise the class can be used independently by setting:
    tau (array of timeslot durations)
    or
    num_tslots and pulse_time for equally spaced timeslots

    Attributes
    ----------
    num_tslots : integer
        Number of timeslots, aka timeslices
        (copied from Dynamics if given)

    pulse_time : float
        total duration of the pulse
        (copied from Dynamics.evo_time if given)

    scaling : float
        linear scaling applied to the pulse
        (copied from Dynamics.initial_ctrl_scaling if given)

    offset : float
        linear offset applied to the pulse
        (copied from Dynamics.initial_ctrl_offset if given)

    tau : array[num_tslots] of float
        Duration of each timeslot
        (copied from Dynamics if given)

    lbound : float
        Lower boundary for the pulse amplitudes
        Note that the scaling and offset attributes can be used to fully
        bound the pulse for all generators except some of the random ones
        This bound (if set) may result in additional shifting / scaling
        Default is -Inf

    ubound : float
        Upper boundary for the pulse amplitudes
        Note that the scaling and offset attributes can be used to fully
        bound the pulse for all generators except some of the random ones
        This bound (if set) may result in additional shifting / scaling
        Default is Inf

    periodic : boolean
        True if the pulse generator produces periodic pulses

    random : boolean
        True if the pulse generator produces random pulses

    """
    def __init__(self, dyn=None):
        self.parent = dyn
        self.reset()

    def reset(self):
        """
        reset attributes to default values
        """
        if isinstance(self.parent, dynamics.Dynamics):
            dyn = self.parent
            self.num_tslots = dyn.num_tslots
            self.pulse_time = dyn.evo_time
            self.scaling = dyn.initial_ctrl_scaling
            self.offset = dyn.initial_ctrl_offset
            self.tau = dyn.tau
        else:
            self.num_tslots = 100
            self.pulse_time = 1.0
            self.scaling = 1.0
            self.tau = None
            self.offset = 0.0

        self._pulse_initialised = False
        self.periodic = False
        self.random = False
        self.lbound = -np.Inf
        self.ubound = np.Inf

    def gen_pulse(self):
        """
        returns the pulse as an array of vales for each timeslot
        Must be implemented by subclass
        """
        # must be implemented by subclass
        raise errors.UsageError(
            "No method defined for generating a pulse. "
            " Suspect base class was used where sub class should have been")

    def init_pulse(self):
        """
        Initialise the pulse parameters
        """
        if self.tau is None:
            self.tau = np.ones(self.num_tslots, dtype='f') * \
                self.pulse_time/self.num_tslots
        self._pulse_initialised = True

        if self.ubound < self.lbound:
            raise ValueError("ubound cannot be less the lbound")

    def _apply_bounds_and_offset(self, pulse):
        """
        Ensure that the randomly generated pulse fits within the bounds
        (after applying the offset)
        Assumes that pulses passed are centered around zero (on average)
        """
        if np.isinf(self.lbound) and np.isinf(self.ubound):
            return pulse + self.offset

        max_amp = max(pulse)
        min_amp = min(pulse)
        if (max_amp + self.offset <= self.ubound and
                min_amp + self.offset >= self.lbound):
            return pulse + self.offset

        # Some shifting / scaling is required.
        bound_range = self.ubound - self.lbound
        if np.isinf(bound_range):
            # One of the bounds is inf, so just shift the pulse
            if np.isinf(self.lbound):
                # max_amp + offset must exceed the ubound
                return pulse + self.ubound - max_amp
            else:
                # min_amp + offset must exceed the lbound
                return pulse + self.lbound - min_amp
        else:
            amp_range = max_amp - min_amp
            if max_amp - min_amp > bound_range:
                # pulse range is too high, it must be scaled
                pulse = pulse * bound_range / amp_range

            # otherwise the pulse should fit anyway
            return pulse + self.lbound - min(pulse)


class PulseGenZero(PulseGen):
    """
    Generates a flat pulse
    """
    def gen_pulse(self):
        """
        Generate a pulse with the same value in every timeslot.
        The value will be zero, unless the offset is not zero,
        in which case it will be the offset
        """
        pulse = np.zeros(self.num_tslots)
        return self._apply_bounds_and_offset(pulse)


class PulseGenRandom(PulseGen):
    """
    Generates random pulses as simply random values for each timeslot
    """
    def reset(self):
        PulseGen.reset(self)
        self.random = True

    def gen_pulse(self):
        """
        Generate a pulse of random values between 1 and -1
        Values are scaled using the scaling property
        and shifted using the offset property
        Returns the pulse as an array of vales for each timeslot
        """
        pulse = (2*np.random.random(self.num_tslots) - 1) * self.scaling

        return self._apply_bounds_and_offset(pulse)


class PulseGenRndFourier(PulseGen):
    """
    Generates pulses by summing sine waves as a Fourier series
    with random coefficients

    Attributes
    ----------
    scaling : float
        The pulses should fit approximately within -/+scaling
        (before the offset is applied)
        as it is used to set a maximum for each component wave
        Use bounds to be sure
        (copied from Dynamics.initial_ctrl_scaling if given)

    min_wavelen : float
        Minimum wavelength of any component wave
        Set by default to 1/10th of the pulse time
    """

    def reset(self):
        """
        reset attributes to default values
        """
        PulseGen.reset(self)
        self.random = True
        try:
            self.min_wavelen = self.pulse_time / 10.0
        except:
            self.min_wavelen = 0.1

    def gen_pulse(self, min_wavelen=None):
        """
        Generate a random pulse based on a Fourier series with a minimum
        wavelength
        """

        if min_wavelen is not None:
            self.min_wavelen = min_wavelen
        min_wavelen = self.min_wavelen

        if min_wavelen > self.pulse_time:
            raise ValueError("Minimum wavelength cannot be greater than "
                             "the pulse time")
        if not self._pulse_initialised:
            self.init_pulse()

        # use some phase to avoid the first pulse being always 0

        sum_wave = np.zeros(self.tau.shape)
        wavelen = 2.0*self.pulse_time

        t = np.zeros(self.num_tslots, dtype=float)
        for k in range(self.num_tslots-1):
            t[k+1] = t[k] + self.tau[k]

        wl = []
        while wavelen > min_wavelen:
            wl.append(wavelen)
            wavelen = wavelen/2.0

        num_comp_waves = len(wl)
        amp_scale = np.sqrt(8)*self.scaling / float(num_comp_waves)

        for wavelen in wl:
            amp = amp_scale*(np.random.rand()*2 - 1)
            phase_off = np.random.rand()*np.pi/2.0
            curr_wave = amp*np.sin(2*np.pi*t/wavelen + phase_off)
            sum_wave += curr_wave

        return self._apply_bounds_and_offset(sum_wave)


class PulseGenRndWaves(PulseGen):
    """
    Generates pulses by summing sine waves with random frequencies
    amplitudes and phase offset

    Attributes
    ----------
    scaling : float
        The pulses should fit approximately within -/+scaling
        (before the offset is applied)
        as it is used to set a maximum for each component wave
        Use bounds to be sure
        (copied from Dynamics.initial_ctrl_scaling if given)

    num_comp_waves : integer
        Number of component waves. That is the number of waves that
        are summed to make the pulse signal
        Set to 20 by default.

    min_wavelen : float
        Minimum wavelength of any component wave
        Set by default to 1/10th of the pulse time

    max_wavelen : float
        Maximum wavelength of any component wave
        Set by default to twice the pulse time
    """

    def reset(self):
        """
        reset attributes to default values
        """
        PulseGen.reset(self)
        self.random = True
        self.num_comp_waves = 20
        try:
            self.min_wavelen = self.pulse_time / 10.0
        except:
            self.min_wavelen = 0.1
        try:
            self.max_wavelen = 2*self.pulse_time
        except:
            self.max_wavelen = 10.0

    def gen_pulse(self, num_comp_waves=None,
                  min_wavelen=None, max_wavelen=None):
        """
        Generate a random pulse by summing sine waves with random freq,
        amplitude and phase offset
        """

        if num_comp_waves is not None:
            self.num_comp_waves = num_comp_waves
        if min_wavelen is not None:
            self.min_wavelen = min_wavelen
        if max_wavelen is not None:
            self.max_wavelen = max_wavelen

        num_comp_waves = self.num_comp_waves
        min_wavelen = self.min_wavelen
        max_wavelen = self.max_wavelen

        if min_wavelen > self.pulse_time:
            raise ValueError("Minimum wavelength cannot be greater than "
                             "the pulse time")
        if max_wavelen <= min_wavelen:
            raise ValueError("Maximum wavelength must be greater than "
                             "the minimum wavelength")

        if not self._pulse_initialised:
            self.init_pulse()

        # use some phase to avoid the first pulse being always 0

        sum_wave = np.zeros(self.tau.shape)

        t = np.zeros(self.num_tslots, dtype=float)
        for k in range(self.num_tslots-1):
            t[k+1] = t[k] + self.tau[k]

        wl_range = max_wavelen - min_wavelen
        amp_scale = np.sqrt(8)*self.scaling / float(num_comp_waves)
        for n in range(num_comp_waves):
            amp = amp_scale*(np.random.rand()*2 - 1)
            phase_off = np.random.rand()*np.pi/2.0
            wavelen = min_wavelen + np.random.rand()*wl_range
            curr_wave = amp*np.sin(2*np.pi*t/wavelen + phase_off)
            sum_wave += curr_wave

        return self._apply_bounds_and_offset(sum_wave)


class PulseGenRndWalk1(PulseGen):
    """
    Generates pulses by using a random walk algorithm

    Attributes
    ----------
    scaling : float
        Used as the range for the starting amplitude
        Note must used bounds if values must be restricted.
        Also scales the max_d_amp value
        (copied from Dynamics.initial_ctrl_scaling if given)

    max_d_amp : float
        Maximum amount amplitude will change between timeslots
        Note this is also factored by the scaling attribute
    """
    def reset(self):
        """
        reset attributes to default values
        """
        PulseGen.reset(self)
        self.random = True
        self.max_d_amp = 0.1

    def gen_pulse(self, max_d_amp=None):
        """
        Generate a pulse by changing the amplitude a random amount between
        -max_d_amp and +max_d_amp at each timeslot. The walk will start at
        a random amplitude between -/+scaling.
        """
        if max_d_amp is not None:
            self.max_d_amp = max_d_amp
        max_d_amp = self.max_d_amp*self.scaling

        if not self._pulse_initialised:
            self.init_pulse()

        walk = np.zeros(self.tau.shape)
        amp = self.scaling*(np.random.rand()*2 - 1)
        for k in range(len(walk)):
            walk[k] = amp
            amp += (np.random.rand()*2 - 1)*max_d_amp

        return self._apply_bounds_and_offset(walk)


class PulseGenRndWalk2(PulseGen):
    """
    Generates pulses by using a random walk algorithm
    Note this is best used with bounds as the walks tend to wander far

    Attributes
    ----------
    scaling : float
        Used as the range for the starting amplitude
        Note must used bounds if values must be restricted.
        Also scales the max_d2_amp value
        (copied from Dynamics.initial_ctrl_scaling if given)

    max_d2_amp : float
        Maximum amount amplitude gradient will change between timeslots
        Note this is also factored by the scaling attribute
    """
    def reset(self):
        """
        reset attributes to default values
        """
        PulseGen.reset(self)
        self.random = True
        self.max_d2_amp = 0.01

    def gen_pulse(self, init_grad_range=None, max_d2_amp=None):
        """
        Generate a pulse by changing the amplitude gradient a random amount
        between -max_d2_amp and +max_d2_amp at each timeslot.
        The walk will start at a random amplitude between -/+scaling.
        The gradient will start at 0
        """
        if max_d2_amp is not None:
            self.max_d2_amp = max_d2_amp

        max_d2_amp = self.max_d2_amp

        if not self._pulse_initialised:
            self.init_pulse()

        walk = np.zeros(self.tau.shape)
        amp = self.scaling*(np.random.rand()*2 - 1)
        print("Start amp {}".format(amp))
        grad = 0.0
        print("Start grad {}".format(grad))
        for k in range(len(walk)):
            walk[k] = amp
            grad += (np.random.rand()*2 - 1)*max_d2_amp
            amp += grad
            # print("grad {}".format(grad))

        return self._apply_bounds_and_offset(walk)


class PulseGenLinear(PulseGen):
    """
    Generates linear pulses

    Attributes
    ----------
    gradient : float
        Gradient of the line.
        Note this is calculated from the start_val and end_val if these
        are given

    start_val : float
        Start point of the line. That is the starting amplitude

    end_val : float
        End point of the line.
        That is the amplitude at the start of the last timeslot
    """

    def reset(self):
        """
        reset attributes to default values
        """
        PulseGen.reset(self)

        self.gradient = None
        self.start_val = -1.0
        self.end_val = 1.0

    def init_pulse(self, gradient=None, start_val=None, end_val=None):
        """
        Calulate the gradient if pulse is defined by start and end point values
        """
        PulseGen.init_pulse(self)
        if start_val is not None and end_val is not None:
            self.start_val = start_val
            self.end_val = end_val

        if self.start_val is not None and self.end_val is not None:
            self.gradient = float(self.end_val - self.start_val) / \
                (self.pulse_time - self.tau[-1])

    def gen_pulse(self, gradient=None, start_val=None, end_val=None):
        """
        Generate a linear pulse using either the gradient and start value
        or using the end point to calulate the gradient
        Note that the scaling and offset parameters are still applied,
        so unless these values are the default 1.0 and 0.0, then the
        actual gradient etc will be different
        Returns the pulse as an array of vales for each timeslot
        """
        if (gradient is not None or
                start_val is not None or end_val is not None):
            self.init_pulse(gradient, start_val, end_val)
        if not self._pulse_initialised:
            self.init_pulse()

        pulse = np.empty(self.num_tslots)
        t = 0.0
        for k in range(self.num_tslots):
            y = self.gradient*t + self.start_val
            pulse[k] = self.scaling*y
            t = t + self.tau[k]

        return self._apply_bounds_and_offset(pulse)


class PulseGenPeriodic(PulseGen):
    """
    Intermediate class for all periodic pulse generators
    All of the periodic pulses range from -1 to 1
    All have a start phase that can be set between 0 and 2pi

    Attributes
    ----------
    num_waves : float
        Number of complete waves (cycles) that occur in the pulse.
        wavelen and freq calculated from this if it is given

    wavelen : float
        Wavelength of the pulse (assuming the speed is 1)
        freq is calculated from this if it is given

    freq : float
        Frequency of the pulse

    start_phase : float
        Phase of the pulse signal when t=0
    """
    def reset(self):
        """
        reset attributes to default values
        """
        PulseGen.reset(self)
        self.periodic = True
        self.num_waves = None
        self.freq = 1.0
        self.wavelen = None
        self.start_phase = 0.0

    def init_pulse(self, num_waves=None, wavelen=None,
                   freq=None, start_phase=None):
        """
        Calculate the wavelength, frequency, number of waves etc
        from the each other and the other parameters
        If num_waves is given then the other parameters are worked from this
        Otherwise if the wavelength is given then it is the driver
        Otherwise the frequency is used to calculate wavelength and num_waves
        """
        PulseGen.init_pulse(self)

        if start_phase is not None:
            self.start_phase = start_phase

        if num_waves is not None or wavelen is not None or freq is not None:
            self.num_waves = num_waves
            self.wavelen = wavelen
            self.freq = freq

        if self.num_waves is not None:
            self.freq = float(self.num_waves) / self.pulse_time
            self.wavelen = 1.0/self.freq
        elif self.wavelen is not None:
            self.freq = 1.0/self.wavelen
            self.num_waves = self.wavelen*self.pulse_time
        else:
            self.wavelen = 1.0/self.freq
            self.num_waves = self.wavelen*self.pulse_time


class PulseGenSine(PulseGenPeriodic):
    """
    Generates sine wave pulses
    """
    def gen_pulse(self, num_waves=None, wavelen=None,
                  freq=None, start_phase=None):
        """
        Generate a sine wave pulse
        If no params are provided then the class object attributes are used.
        If they are provided, then these will reinitialise the object attribs.
        returns the pulse as an array of vales for each timeslot
        """
        if start_phase is not None:
            self.start_phase = start_phase

        if num_waves is not None or wavelen is not None or freq is not None:
            self.init_pulse(num_waves, wavelen, freq, start_phase)

        if not self._pulse_initialised:
            self.init_pulse()

        pulse = np.empty(self.num_tslots)
        t = 0.0
        for k in range(self.num_tslots):
            phase = 2*np.pi*self.freq*t + self.start_phase
            pulse[k] = self.scaling*np.sin(phase)
            t = t + self.tau[k]
        return self._apply_bounds_and_offset(pulse)


class PulseGenSquare(PulseGenPeriodic):
    """
    Generates square wave pulses
    """
    def gen_pulse(self, num_waves=None, wavelen=None,
                  freq=None, start_phase=None):
        """
        Generate a square wave pulse
        If no parameters are pavided then the class object attributes are used.
        If they are provided, then these will reinitialise the object attribs
        """
        if start_phase is not None:
            self.start_phase = start_phase

        if num_waves is not None or wavelen is not None or freq is not None:
            self.init_pulse(num_waves, wavelen, freq, start_phase)

        if not self._pulse_initialised:
            self.init_pulse()

        pulse = np.empty(self.num_tslots)
        t = 0.0
        for k in range(self.num_tslots):
            phase = 2*np.pi*self.freq*t + self.start_phase
            x = phase/(2*np.pi)
            y = 4*np.floor(x) - 2*np.floor(2*x) + 1
            pulse[k] = self.scaling*y
            t = t + self.tau[k]
        return self._apply_bounds_and_offset(pulse)


class PulseGenSaw(PulseGenPeriodic):
    """
    Generates saw tooth wave pulses
    """
    def gen_pulse(self, num_waves=None, wavelen=None,
                  freq=None, start_phase=None):
        """
        Generate a saw tooth wave pulse
        If no parameters are pavided then the class object attributes are used.
        If they are provided, then these will reinitialise the object attribs
        """
        if start_phase is not None:
            self.start_phase = start_phase

        if num_waves is not None or wavelen is not None or freq is not None:
            self.init_pulse(num_waves, wavelen, freq, start_phase)

        if not self._pulse_initialised:
            self.init_pulse()

        pulse = np.empty(self.num_tslots)
        t = 0.0
        for k in range(self.num_tslots):
            phase = 2*np.pi*self.freq*t + self.start_phase
            x = phase/(2*np.pi)
            y = 2*(x - np.floor(0.5 + x))
            pulse[k] = self.scaling*y
            t = t + self.tau[k]
        return self._apply_bounds_and_offset(pulse)


class PulseGenTriangle(PulseGenPeriodic):
    """
    Generates triangular wave pulses
    """
    def gen_pulse(self, num_waves=None, wavelen=None,
                  freq=None, start_phase=None):
        """
        Generate a sine wave pulse
        If no parameters are pavided then the class object attributes are used.
        If they are provided, then these will reinitialise the object attribs
        """
        if start_phase is not None:
            self.start_phase = start_phase

        if num_waves is not None or wavelen is not None or freq is not None:
            self.init_pulse(num_waves, wavelen, freq, start_phase)

        if not self._pulse_initialised:
            self.init_pulse()

        pulse = np.empty(self.num_tslots)
        t = 0.0
        for k in range(self.num_tslots):
            phase = 2*np.pi*self.freq*t + self.start_phase
            x = phase/(2*np.pi)
            y = 2*np.abs(2*(x - np.floor(0.5 + x))) - 1
            pulse[k] = self.scaling*y
            t = t + self.tau[k]

        return self._apply_bounds_and_offset(pulse)
