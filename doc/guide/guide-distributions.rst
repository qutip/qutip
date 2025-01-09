.. _distributions:

**********************************************
Probability distributions in quantum mechanics
**********************************************

Probability distributions in quantum mechanics are fundamental and intrinsic to the nature of quantum systems. Encoded in the wave function, the probabilities are obtained by the Born rule : :math:`P(x)= |\psi(x)|²`. Their are many distributions, some of them that users of QuTiP can generate and use in their project.

.. _harmonic-oscillator:

The quantum harmonic oscillator
===============================

The quantification of the classical harmonic oscillator yields the Hamiltonian :

.. math::

    \displaystyle H = \frac{1}{2m}\hat{P}² + \frac{m \omega²}{2}X²

where :math:`m` is the mass of the particule, :math:`\omega` is the angular frequency of the oscillator, :math:`\hat{P}` is the momentum operator and :math:`\hat{X}` is the position operator.

Define the ladder operators

.. math::

    \displaystyle a = \sqrt{\frac{m \omega}{2 \hbar}} \left(\hat{X} + \frac{i}{m \omega}\hat{P}\right)

and 

.. math::

    \displaystyle \dagger{a} = \sqrt{\frac{m \omega}{2 \hbar}} \left(\hat{X} - \frac{i}{m \omega}\hat{P}\right)

Then, the Hamiltonian can be written in terms of the ladder operators as

.. math::

    \displaystyle H = \hbar \omega \left(\dagger{a}a + \frac{1}{2}I\right) = \hbar \omega \left(\hat{N} + \frac{1}{2}I\right) 

where :math:`\hat{N}` is the number operator.

It can be shown that on an eigenstate :math:`|n>` of the number operator, the ladder operators accomplish

.. math::

    \displaystyle a |n> = \sqrt{n}|n-1> 
.. math::

    \displaystyle \dagger{a} |n> = \sqrt{n+1}|n+1> 

From that, it is easy to see the equation for all energy levels of the harmonic oscillator.

.. math::

    \displaystyle E_n = \hbar \omega \left(n + \frac{1}{2}\right)

To find the wave function :math:`\psi(x)_0` of the ground state (:math:`n=0`), we have to compute :math:`<x|0>`. Knowing that :math:`a|0> = 0`, we have 

.. math::

    \displaystyle <x|a|0> = \sqrt{\frac{m \omega}{2 \hbar}} \left[<x|\hat{X}|0> + \frac{i}{m \omega} <x|\hat{P}|0>\right] = \sqrt{\frac{m \omega}{2 \hbar}} \left[x\psi(x)_0 + \frac{\hbar}{m \omega} \frac{\partial}{\partial x}\psi(x)_0\right] = 0

from the fact that in the coordinate basis

.. math::

    \displaystyle \hat{P} = -i\hbar \frac{\partial}{\partial x}

This implies that

.. math::

    \displaystyle \frac{-m \omega}{\hbar}x = \frac{1}{\psi(x)_0}\frac{\partial}{\partial x}\psi(x)_0

with a single solution for the ground state giving

.. math::

    \displaystyle \psi(x)_0 = \left(\frac{m \omega}{\pi \hbar}\right)^{\frac{1}{4}} e^{-\frac{m \omega}{2\hbar}x²}

By the Born rule, we obtain the probability distribution for the ground state. We can use the class `HarmonicOscillatorProbabilityFunction` from `qutip.distributions` in order to visualize what the distribution would look like for any state of the harmonic oscillator (see qutip-tutorials/tutorials-v5/visualization/distributions.md). Here, we would have all distributions from :math:`n=0` to :math:`n=7`.

.. code-block:: Python

   from qutip import *
   from qutip.distributions import *
   import matplotlib.pyplot as plt

   M=8
   N=20

   fig, ax = plt.subplots(M, 1, figsize=(10, 12), sharex=True)

   for n in range(M):
       psi = fock(N, n)
       wf = HarmonicOscillatorProbabilityFunction(psi, 1.0, extent=[-10, 10])
       wf.visualize(fig=fig, ax=ax[M-n-1], show_ylabel=False, show_xlabel=(n == 0))

Other distributions
===================

There are other distributions like `WignerDistribution`, `QDistribution` or `TwoModeQuadratureCorrelation` that can be used with qutip (again, see qutip-tutorials/tutorials-v5/visualization/distributions.md for examples).

