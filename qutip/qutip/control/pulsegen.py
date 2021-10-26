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

import qutip.logging_utils as logging
logger = logging.get_logger()

import qutip.control.dynamics as dynamics
import qutip.control.errors as errors

def create_pulse_gen(pulse_type='RND', dyn=None, pulse_params=None):
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
        return PulseGenRandom(dyn, params=pulse_params)
    if pulse_type == 'RNDFOURIER':
        return PulseGenRndFourier(dyn, params=pulse_params)
    if pulse_type == 'RNDWAVES':
        return PulseGenRndWaves(dyn, params=pulse_params)
    if pulse_type == 'RNDWALK1':
        return PulseGenRndWalk1(dyn, params=pulse_params)
    if pulse_type == 'RNDWALK2':
        return PulseGenRndWalk2(dyn, params=pulse_params)
    elif pulse_type == 'LIN':
        return PulseGenLinear(dyn, params=pulse_params)
    elif pulse_type == 'ZERO':
        return PulseGenZero(dyn, params=pulse_params)
    elif pulse_type == 'SINE':
        return PulseGenSine(dyn, params=pulse_params)
    elif pulse_type == 'SQUARE':
        return PulseGenSquare(dyn, params=pulse_params)
    elif pulse_type == 'SAW':
        return PulseGenSaw(dyn, params=pulse_params)
    elif pulse_type == 'TRIANGLE':
        return PulseGenTriangle(dyn, params=pulse_params)
    elif pulse_type == 'GAUSSIAN':  
        return PulseGenGaussian(dyn, params=pulse_params)
    elif pulse_type == 'CRAB_FOURIER':
        return PulseGenCrabFourier(dyn, params=pulse_params)
    elif pulse_type == 'GAUSSIAN_EDGE':  
        return PulseGenGaussianEdge(dyn, params=pulse_params)
    else:
        raise ValueError("No option for pulse_type '{}'".format(pulse_type))


class PulseGen(object):
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

    log_level : integer
        level of messaging output from the logger.
        Options are attributes of qutip.logging_utils,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN
    """
    def __init__(self, dyn=None, params=None):
        self.parent = dyn
        self.params = params
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
            self.log_level = dyn.log_level
        else:
            self.num_tslots = 100
            self.pulse_time = 1.0
            self.scaling = 1.0
            self.tau = None
            self.offset = 0.0

        self._uses_time = False
        self.time = None
        self._pulse_initialised = False
        self.periodic = False
        self.random = False
        self.lbound = None
        self.ubound = None
        self.ramping_pulse = None
        
        self.apply_params()
        
    def apply_params(self, params=None):
        """
        Set object attributes based on the dictionary (if any) passed in the 
        instantiation, or passed as a parameter
        This is called during the instantiation automatically.
        The key value pairs are the attribute name and value
        """               
        if not params:
            params = self.params
        
        if isinstance(params, dict):
            self.params = params
            for key in params:
                setattr(self, key, params[key])

    @property
    def log_level(self):
        return logger.level        
        
    @log_level.setter
    def log_level(self, lvl):
        """
        Set the log_level attribute and set the level of the logger
        that is call logger.setLevel(lvl)
        """
        logger.setLevel(lvl)
        
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
                
        if self._uses_time:
            self.time = np.zeros(self.num_tslots, dtype=float)
            for k in range(self.num_tslots-1):
                self.time[k+1] = self.time[k] + self.tau[k]
                
        self._pulse_initialised = True

        if not self.lbound is None:
            if np.isinf(self.lbound):
                self.lbound = None
        if not self.ubound is None:
            if np.isinf(self.ubound):
                self.ubound = None
        
        if not self.ubound is None and not self.lbound is None:
            if self.ubound < self.lbound:
                raise ValueError("ubound cannot be less the lbound")

    def _apply_bounds_and_offset(self, pulse):
        """
        Ensure that the randomly generated pulse fits within the bounds
        (after applying the offset)
        Assumes that pulses passed are centered around zero (on average)
        """
        if self.lbound is None and self.ubound is None:
            return pulse + self.offset

        max_amp = max(pulse)
        min_amp = min(pulse)
        if ((self.ubound is None or max_amp + self.offset <= self.ubound) and
            (self.lbound is None or min_amp + self.offset >= self.lbound)):
            return pulse + self.offset

        # Some shifting / scaling is required.
        if self.ubound is None or self.lbound is None:
            # One of the bounds is inf, so just shift the pulse
            if self.lbound is None:
                # max_amp + offset must exceed the ubound
                return pulse + self.ubound - max_amp
            else:
                # min_amp + offset must exceed the lbound
                return pulse + self.lbound - min_amp
        else:
            bound_range = self.ubound - self.lbound
            amp_range = max_amp - min_amp
            if max_amp - min_amp > bound_range:
                # pulse range is too high, it must be scaled
                pulse = pulse * bound_range / amp_range

            # otherwise the pulse should fit anyway
            return pulse + self.lbound - min(pulse)

    def _apply_ramping_pulse(self, pulse, ramping_pulse=None):
        if ramping_pulse is None:
            ramping_pulse = self.ramping_pulse
        if ramping_pulse is not None:
            pulse = pulse*ramping_pulse
            
        return pulse
        
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
        self.apply_params()

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
        self._uses_time = True
        try:
            self.min_wavelen = self.pulse_time / 10.0
        except:
            self.min_wavelen = 0.1
        self.apply_params()

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

        t = self.time
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
        self._uses_time = True
        self.num_comp_waves = 20
        try:
            self.min_wavelen = self.pulse_time / 10.0
        except:
            self.min_wavelen = 0.1
        try:
            self.max_wavelen = 2*self.pulse_time
        except:
            self.max_wavelen = 10.0
        self.apply_params()

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

        t = self.time
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
        self.apply_params()

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
        self.apply_params()

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
        self.apply_params()

    def init_pulse(self, gradient=None, start_val=None, end_val=None):
        """
        Calculate the gradient if pulse is defined by start and 
        end point values
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
        self.apply_params()

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
            phase = 2*np.pi*self.freq*t + self.start_phase + np.pi/2.0
            x = phase/(2*np.pi)
            y = 2*np.abs(2*(x - np.floor(0.5 + x))) - 1
            pulse[k] = self.scaling*y
            t = t + self.tau[k]

        return self._apply_bounds_and_offset(pulse)
        
class PulseGenGaussian(PulseGen):
    """
    Generates pulses with a Gaussian profile
    """
    def reset(self):
        """
        reset attributes to default values
        """
        PulseGen.reset(self)
        self._uses_time = True
        self.mean = 0.5*self.pulse_time
        self.variance = 0.5*self.pulse_time
        self.apply_params()
        
    def gen_pulse(self, mean=None, variance=None):
        """
        Generate a pulse with Gaussian shape. The peak is centre around the
        mean and the variance determines the breadth
        The scaling and offset attributes are applied as an amplitude
        and fixed linear offset. Note that the maximum amplitude will be
        scaling + offset.
        """
        if not self._pulse_initialised:
            self.init_pulse()
        
        if mean:
            Tm = mean
        else:
            Tm = self.mean
        if variance:
            Tv = variance
        else:
            Tv = self.variance
        t = self.time
        T = self.pulse_time

        pulse = self.scaling*np.exp(-(t-Tm)**2/(2*Tv))
        return self._apply_bounds_and_offset(pulse)

class PulseGenGaussianEdge(PulseGen):
    """
    Generate pulses with inverted Gaussian ramping in and out
    It's intended use for a ramping modulation, which is often required in 
    experimental setups.
    
    Attributes
    ----------
        decay_time : float
            Determines the ramping rate. It is approximately the time
            required to bring the pulse to full amplitude
            It is set to 1/10 of the pulse time by default
    """

    def reset(self):
        """
        reset attributes to default values
        """
        PulseGen.reset(self)
        self._uses_time = True
        self.decay_time = self.pulse_time / 10.0
        self.apply_params()

    def gen_pulse(self, decay_time=None):
        """
        Generate a pulse that starts and ends at zero and 1.0 in between
        then apply scaling and offset
        The tailing in and out is an inverted Gaussian shape
        """
        if not self._pulse_initialised:
            self.init_pulse()
            
        t = self.time
        if decay_time:
            Td = decay_time
        else:
            Td = self.decay_time
        T = self.pulse_time
        pulse = 1.0 - np.exp(-t**2/Td) - np.exp(-(t-T)**2/Td)
        pulse = pulse*self.scaling

        return self._apply_bounds_and_offset(pulse)


### The following are pulse generators for the CRAB algorithm ###
# AJGP 2015-05-14: 
# The intention is to have a more general base class that allows
# setting of general basis functions

class PulseGenCrab(PulseGen):
    """
    Base class for all CRAB pulse generators
    Note these are more involved in the optimisation process as they are
    used to produce piecewise control amplitudes each time new optimisation
    parameters are tried
    
    Attributes
    ----------
    num_coeffs : integer
        Number of coefficients used for each basis function
        
    num_basis_funcs : integer
        Number of basis functions
        In this case set at 2 and should not be changed
        
    coeffs : float array[num_coeffs, num_basis_funcs]
        The basis coefficient values
        
    randomize_coeffs : bool
        If True (default) then the coefficients are set to some random values
        when initialised, otherwise they will all be equal to self.scaling
    """
    def __init__(self, dyn=None, num_coeffs=None, params=None):
        self.parent = dyn
        self.num_coeffs = num_coeffs
        self.params = params
        self.reset()
        
    def reset(self):
        """
        reset attributes to default values
        """
        PulseGen.reset(self)
        self.NUM_COEFFS_WARN_LVL = 20
        self.DEF_NUM_COEFFS = 4
        self._BSC_ALL = 1
        self._BSC_GT_MEAN = 2
        self._BSC_LT_MEAN = 3
        
        self._uses_time = True
        self.time = None
        self.num_basis_funcs = 2
        self.num_optim_vars = 0
        self.coeffs = None
        self.randomize_coeffs = True
        self._num_coeffs_estimated = False
        self.guess_pulse_action = 'MODULATE'
        self.guess_pulse = None
        self.guess_pulse_func = None
        self.apply_params()
        
    def init_pulse(self, num_coeffs=None):
        """
        Set the initial freq and coefficient values
        """
        PulseGen.init_pulse(self)
        self.init_coeffs(num_coeffs=num_coeffs)
        
        if self.guess_pulse is not None:
            self.init_guess_pulse()
        self._init_bounds()
        
        if self.log_level <= logging.DEBUG and not self._num_coeffs_estimated:
            logger.debug(
                    "CRAB pulse initialised with {} coefficients per basis "
                    "function, which means a total of {} "
                    "optimisation variables for this pulse".format(
                            self.num_coeffs, self.num_optim_vars))
        
#    def generate_guess_pulse(self)
#        if isinstance(self.guess_pulsegen, PulseGen):
#            self.guess_pulse = self.guess_pulsegen.gen_pulse()
#        return self.guess_pulse
        
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
        
        self._num_coeffs_estimated = False
        if not self.num_coeffs:
            if isinstance(self.parent, dynamics.Dynamics):
                dim = self.parent.get_drift_dim()
                self.num_coeffs = self.estimate_num_coeffs(dim)
                self._num_coeffs_estimated = True
            else:
                self.num_coeffs = self.DEF_NUM_COEFFS
        self.num_optim_vars = self.num_coeffs*self.num_basis_funcs
        
        if self._num_coeffs_estimated:
            if self.log_level <= logging.INFO:
                logger.info(
                    "The number of CRAB coefficients per basis function "
                    "has been estimated as {}, which means a total of {} "
                    "optimisation variables for this pulse. Based on the "
                    "dimension ({}) of the system".format(
                            self.num_coeffs, self.num_optim_vars, dim))
            # Issue warning if beyond the recommended level
            if self.log_level <= logging.WARN:
                if self.num_coeffs > self.NUM_COEFFS_WARN_LVL:
                    logger.warn(
                        "The estimated number of coefficients {} exceeds "
                        "the amount ({}) recommended for efficient "
                        "optimisation. You can set this level explicitly "
                        "to suppress this message.".format(
                            self.num_coeffs, self.NUM_COEFFS_WARN_LVL))
                            
        if self.randomize_coeffs:
            r = np.random.random([self.num_coeffs, self.num_basis_funcs])
            self.coeffs = (2*r - 1.0) * self.scaling
        else:
            self.coeffs = np.ones([self.num_coeffs, 
                                   self.num_basis_funcs])*self.scaling
        
    def estimate_num_coeffs(self, dim):
        """
        Estimate the number coefficients based on the dimensionality of the
        system.
        Returns
        -------
        num_coeffs : int
            estimated number of coefficients
        """
        num_coeffs = max(2, dim - 1)
        return num_coeffs
        
    def get_optim_var_vals(self):
        """
        Get the parameter values to be optimised
        Returns
        -------
        list (or 1d array) of floats 
        """
        return self.coeffs.ravel().tolist()
    
    def set_optim_var_vals(self, param_vals):
        """
        Set the values of the any of the pulse generation parameters
        based on new values from the optimisation method
        Typically this will be the basis coefficients
        """
        # Type and size checking avoided here as this is in the 
        # main optmisation call sequence
        self.set_coeffs(param_vals)
        
    def set_coeffs(self, param_vals):
        self.coeffs = param_vals.reshape(
                    [self.num_coeffs, self.num_basis_funcs])
    
    def init_guess_pulse(self):
        
        self.guess_pulse_func = None
        if not self.guess_pulse_action:
            logger.WARN("No guess pulse action given, hence ignored.")
        elif self.guess_pulse_action.upper() == 'MODULATE':
            self.guess_pulse_func = self.guess_pulse_modulate
        elif self.guess_pulse_action.upper() == 'ADD':
            self.guess_pulse_func = self.guess_pulse_add
        else:
            logger.WARN("No option for guess pulse action '{}' "
                        ", hence ignored.".format(self.guess_pulse_action))
    
    def guess_pulse_add(self, pulse):
        pulse = pulse + self.guess_pulse
        return pulse
        
    def guess_pulse_modulate(self, pulse):
        pulse = (1.0 + pulse)*self.guess_pulse
        return pulse
        
    def _init_bounds(self):
        add_guess_pulse_scale = False
        if self.lbound is None and self.ubound is None:
            # no bounds to apply
            self._bound_scale_cond = None
        elif self.lbound is None:
            # only upper bound
            if self.ubound > 0:
                self._bound_mean = 0.0
                self._bound_scale = self.ubound
            else:
                add_guess_pulse_scale = True
                self._bound_scale = self.scaling*self.num_coeffs + \
                            self.get_guess_pulse_scale()
                self._bound_mean = -abs(self._bound_scale) + self.ubound
            self._bound_scale_cond = self._BSC_GT_MEAN

        elif self.ubound is None:
            # only lower bound
            if self.lbound < 0:
                self._bound_mean = 0.0
                self._bound_scale = abs(self.lbound)
            else:
                self._bound_scale = self.scaling*self.num_coeffs + \
                            self.get_guess_pulse_scale()
                self._bound_mean = abs(self._bound_scale) + self.lbound
            self._bound_scale_cond = self._BSC_LT_MEAN

        else:
            # lower and upper bounds
            self._bound_mean = 0.5*(self.ubound + self.lbound)
            self._bound_scale = 0.5*(self.ubound - self.lbound)
            self._bound_scale_cond = self._BSC_ALL
            
    def get_guess_pulse_scale(self):
        scale = 0.0
        if self.guess_pulse is not None:
            scale = max(np.amax(self.guess_pulse) - np.amin(self.guess_pulse),
                        np.amax(self.guess_pulse))
        return scale
        
    def _apply_bounds(self, pulse):
        """
        Scaling the amplitudes using the tanh function if there are bounds
        """
        if self._bound_scale_cond == self._BSC_ALL:
            pulse = np.tanh(pulse)*self._bound_scale + self._bound_mean
            return pulse
        elif self._bound_scale_cond == self._BSC_GT_MEAN:
            scale_where = pulse > self._bound_mean
            pulse[scale_where] = (np.tanh(pulse[scale_where])*self._bound_scale
                                        + self._bound_mean)
            return pulse
        elif self._bound_scale_cond == self._BSC_LT_MEAN:
            scale_where = pulse < self._bound_mean
            pulse[scale_where] = (np.tanh(pulse[scale_where])*self._bound_scale
                                        + self._bound_mean)
            return pulse
        else:
            return pulse
       

class PulseGenCrabFourier(PulseGenCrab):
    """
    Generates a pulse using the Fourier basis functions, i.e. sin and cos

    Attributes
    ----------
    freqs : float array[num_coeffs]
        Frequencies for the basis functions
    randomize_freqs : bool
        If True (default) the some random offset is applied to the frequencies
    """

    def reset(self):
        """
        reset attributes to default values
        """
        PulseGenCrab.reset(self)
        self.freqs = None
        self.randomize_freqs = True

    def init_pulse(self, num_coeffs=None):
        """
        Set the initial freq and coefficient values
        """
        PulseGenCrab.init_pulse(self)
            
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
        
        if self.randomize_freqs:
            self.freqs += np.random.random(self.num_coeffs) - 0.5
        
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
            phase = self.freqs[i]*self.time
#            basis1comp = self.coeffs[i, 0]*np.sin(phase)
#            basis2comp = self.coeffs[i, 1]*np.cos(phase)
#            pulse += basis1comp + basis2comp
            pulse += self.coeffs[i, 0]*np.sin(phase) + \
                        self.coeffs[i, 1]*np.cos(phase) 

        if self.guess_pulse_func:
            pulse = self.guess_pulse_func(pulse)
        if self.ramping_pulse is not None:
            pulse = self._apply_ramping_pulse(pulse)
            
        return self._apply_bounds(pulse)
    