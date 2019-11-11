"""
Basic use of Processor
=============================
 
This example contains the basic functions of :class:`qutip.qip.device.Processor.` We define a simulator with control Hamiltonian, pulse amplitude and time slice for each pulse. The two figures illustrate the pulse shape for two different setup: step function or continuous pulse.
"""
import copy
import numpy as np
import matplotlib.pyplot as plt
pi = np.pi
from qutip.qip.device import Processor
from qutip.operators import sigmaz
from qutip.states import basis

processor = Processor(N=1)
processor.add_ctrl(sigmaz(), targets=0)

tlist = np.linspace(0., 2*np.pi, 20)
processor = Processor(N=1, spline_kind="step_func")
processor.add_ctrl(sigmaz())
processor.tlist = tlist
processor.coeffs = np.array([[np.sin(t) for t in tlist]])
processor.plot_pulses(noisy=False)

tlist = np.linspace(0., 2*np.pi, 20)
processor = Processor(N=1, spline_kind="cubic")
processor.add_ctrl(sigmaz())
processor.tlist = tlist
processor.coeffs = np.array([[np.sin(t) for t in tlist]])
processor.plot_pulses(noisy=False)
