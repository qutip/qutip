"""
Control Amplitude Noise
=======================

This example demonstrates how to add Gaussian noise to the control pulse.
"""
import numpy as np
import matplotlib.pyplot as plt
from qutip.qip.device import Processor
from qutip.qip import RandomNoise
from qutip.operators import sigmaz, sigmay

# add control Hamiltonians
processor = Processor(N=1)
processor.add_ctrl(sigmaz(), targets=0)
processor.add_ctrl(sigmay(), targets=0)

# define coeffs and tlist
processor.coeffs = np.array([[ 0.3, 0.,  0.2],
                            [ 0. , 0.5, 0. ]])
processor.tlist = np.array([0., np.pi/2., 2*np.pi/2, 3*np.pi/2])

# define noise, loc and scale are keyword arguments for np.random.normal
processor.add_noise(RandomNoise(dt=0.1, loc=0.08, scale=0.02, ))
processor.plot_pulses(noisy=False, title="Original control amplitude", figsize=(5,3))
processor.plot_pulses(noisy=True, title="Noisy control amplitude", figsize=(5,3))
