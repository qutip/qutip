.. _distributions:

**********************************************
Probability distributions in quantum mechanics
**********************************************

Probability distributions in quantum mechanics are fundamental and intrinsic to the nature of quantum systems. Encoded in the wave function, the probabilities are obtained by the Born rule : :math:`P(x)= |\psi(x)|²`. Their are many distributions, some of them that users of QuTiP can generate and use in their project. Let's see in details one of such available distributions.

.. _harmonic-oscillator:

The quantum harmonic oscillator
===============================

Probably the easiest probability distribution to show is the one for the quantum harmonic oscillator. The quantification of the classical harmonic oscillator yields the Hamiltonian :

.. math::

    \displaystyle \hat{H} = \frac{1}{2m}\hat{P}² + \frac{m \omega²}{2}\hat{X}²

where :math:`m` is the mass of the particule, :math:`\omega` is the angular frequency of the oscillator, :math:`\hat{P}` is the momentum operator and :math:`\hat{X}` is the position operator. Define the ladder operators

.. math::

    \displaystyle \hat{a} = \sqrt{\frac{m \omega}{2 \hbar}} \left(\hat{X} + \frac{i}{m \omega}\hat{P}\right)

and 

.. math::

    \displaystyle \hat{a}^\dagger = \sqrt{\frac{m \omega}{2 \hbar}} \left(\hat{X} - \frac{i}{m \omega}\hat{P}\right)

Then, the Hamiltonian can be written in terms of the ladder operators as

.. math::

    \displaystyle \hat{H} = \hbar \omega \left(\hat{a}^\dagger \hat{a} + \frac{1}{2}\hat{I}\right) = \hbar \omega \left(\hat{N} + \frac{1}{2}\hat{I}\right) 

where :math:`\hat{N}` is the number operator. It can be shown that on an eigenstate :math:`|n>` of the number operator (:math:`n \geq 0`), the ladder operators accomplish

.. math::

    \displaystyle \hat{a} |n> = \sqrt{n}|n-1> 
.. math::

    \displaystyle \hat{a}^\dagger |n> = \sqrt{n+1}|n+1> 

From that, it is easy to see that energy levels of the harmonic oscillator obey the following :

.. math::

    \displaystyle E_n = \hbar \omega \left(n + \frac{1}{2}\right)

.. _wave_function:

Finding the wave function of the ground state
=============================================

To find the wave function :math:`\psi_0(x)` of the ground state (:math:`n=0`), we have to compute :math:`<x|0>`. Knowing that :math:`\hat{a}|0> = 0`, we have 

.. math::

    \displaystyle <x|\hat{a}|0> = \sqrt{\frac{m \omega}{2 \hbar}} \left[<x|\hat{X}|0> + \frac{i}{m \omega} <x|\hat{P}|0>\right] = \sqrt{\frac{m \omega}{2 \hbar}} \left[x\psi_0(x) + \frac{\hbar}{m \omega} \frac{\partial}{\partial x}\psi_0(x)\right] = 0

from the fact that in the coordinate basis

.. math::

    \displaystyle \hat{P} = -i\hbar \frac{\partial}{\partial x}

This implies that

.. math::

    \displaystyle \frac{-m \omega}{\hbar}x = \frac{1}{\psi_0(x)}\frac{\partial}{\partial x}\psi_0(x)

with a single solution for the ground state giving

.. math::

    \displaystyle \psi_0(x) = \left(\frac{m \omega}{\pi \hbar}\right)^{1/4} e^{-\frac{m \omega}{2\hbar}x²}

By the Born rule, we obtain the probability distribution for the ground state. Because all states can be reached by the ladder operators from the ground state, one can generalize the wave function and probability distribution for the state :math:`|n>`.

.. _implementation:

QuTiP implementation
====================

We can use the classes `HarmonicOscillatorWaveFunction` and `HarmonicOscillatorProbabilityFunction` from `qutip.distributions` in order to visualize what the wave function and probability distribution would look like for any state of the harmonic oscillator. Here, we would have all wave functions followed by all probability distributions from :math:`n=0` to :math:`n=7`.

.. code-block:: Python
    
    from qutip import fock
    from qutip.distributions import HarmonicOscillatorWaveFunction
    import matplotlib.pyplot as plt

    M=8
    N=20

    fig, ax = plt.subplots(M, 1, figsize=(10, 12), sharex=True)

    for n in range(M):
        psi = fock(N, n)
        wf = HarmonicOscillatorWaveFunction(psi, 1.0, extent=[-10, 10])
        wf.visualize(fig=fig, ax=ax[M-n-1], show_ylabel=False, show_xlabel=(n == 0))

.. code-block:: Python
    
    from qutip import fock
    from qutip.distributions import HarmonicOscillatorProbabilityFunction
    import matplotlib.pyplot as plt

    M=8
    N=20

    fig, ax = plt.subplots(M, 1, figsize=(10, 12), sharex=True)

    for n in range(M):
        psi = fock(N, n)
        prob = HarmonicOscillatorProbabilityFunction(psi, 1.0, extent=[-10, 10])
        prob.visualize(fig=fig, ax=ax[M-n-1], show_ylabel=False, show_xlabel=(n == 0))