.. _dysolve:

*******
Dysolve
*******

The following guide explains what Dysolve [1]_ is and how to use its implementation in QuTiP.

Time evolution of a quantum system
==================================
It is common knowledge that the time evolution of a closed quantum system is given by the Schrödinger equation (:math:`\hbar = 1`) :

.. math::

 \displaystyle i \frac{d}{dt}\left|\psi(t)\right> = H(t)\left|\psi(t)\right>

When solving this equation, a time evolution operator :math:`U(t_f, t_i)` (often called a propagator) is introduced and allows a quantum state to evolve from time :math:`t_i` to time :math:`t_f` when applied. When the hamiltonian is time dependent, the propagator has the following form [2]_ : 

.. math::

 \displaystyle U(t_f, t_i) = e^{-i\int_{t_i}^{t_f}H(t)dt}

There are many ways to compute such a propagator with code depending on what the system's hamiltonian is.

.. _DysolvePropagator:

Calculating propagators with Dysolve
====================================

For hamiltonians of the form :math:`H(t) = H_0 + \cos(\omega t)X` where a perturbation :math:`X` is added to some quantum system with hamiltonian :math:`H_0`, Dysolve is an excellent method to compute time propagators. Indeed, the calculations can be done in parallel and certain quantities can be reused throughout the whole process. Futhermore, Dysolve can be generalized to more complicated perturbations, offering a wide range of possible applications. Also, because this method is specifically designed for oscillating perturbations, it gives more precise results quicker compared to more general methods. This effect is accentuated the bigger the frequency is.

After plugging in :math:`H(t) = H_0 + \cos(\omega t)X` in the equation for the propagator, the idea is to develop the exponential as a power series of :math:`X`. Furthermore, the cosine is rewritten using its exponential equivalent. From that, a propagator will consist of a sum of subpropagators :math:`U^{(n)}`

.. math::

 \displaystyle U(t+\delta t,t) = \sum_{n=0}^{\infty} U^{(n)}(t + \delta t, t)

where 

.. math::

 \displaystyle U^{(n)}(t + \delta t, t) = \sum_{\left\{\boldsymbol{\omega}_n\right\}}e^{i\sum_{p=1}^{n}\boldsymbol{\omega}_n[p]t}S^{(n)}(\boldsymbol{\omega}_n, \delta t)

Here, :math:`\{\boldsymbol{\omega}_n\}` is the set of vectors of length :math:`n` of the form :math:`\boldsymbol{\omega}_n = \left[±\omega, ..., ±\omega\right]`, :math:`S^{(n)}(\boldsymbol{\omega}_n, \delta t)` is a matrix and it is assumed that :math:`H_0` and :math:`X` are written in :math:`H_0`'s basis. In practice, the summation over :math:`n` is truncated to the first few terms and the evolution is divided into small time increments. So, with :math:`\tau` being the time ordering operator,

.. math::
 \displaystyle U(T,0) = \tau\prod_{p=0}^{P}U((p+1)\delta t, p\delta t) = \tau\prod_{p=0}^{P}\left(\sum_{n=0}^{\infty}U^{(n)}((p+1)\delta t, p\delta t)\right)

.. math::
 \displaystyle \approx  \tau\prod_{p=0}^{P}\left(\sum_{n=0}^{r}U^{(n)}((p+1)\delta t, p\delta t)\right)

As we can see, computing :math:`S^{(n)}(\boldsymbol{\omega}_n, \delta t)` for a given range on :math:`n` once allows us to calculate as many propagators as we want because this quantity depends on the time increment :math:`\delta t` and not the current time in the evolution. Therefore, in the long run and with the right ressources, Dysolve can be an extremely useful and efficient method for calculating propagators of systems with the form described at the beginning of this section.

.. _dysolve_code_example:

Code example
============

The following code shows a simple example on how to use QuTiP's implementation of Dysolve. Here, :math:`H(t) = \sigma_z + \cos(t)\sigma_x`.

.. code-block:: Python
    
    from qutip.solver.dysolve_propagator import DysolvePropagator, dysolve_propagator
    from qutip import sigmax, sigmaz

    H_0 = sigmaz()
    X = sigmax()
    omega = 1
    options = {'max_order': 5, 'max_dt': 0.05}

    #Initialize an instance and call it to compute a time propagator.
    #Give a final time and, optionally, an initial time.
    t_f = 1
    dy = DysolvePropagator(H_0, X, omega, options)
    U = dy(t_f)
    
    #Another option is to use the dysolve_propagator function.
    #A final time or a list of times can be given.
    #For the latter, [U(t[i], t[0])] is returned.
    times = [0, 1, 2]
    Us = dysolve_propagator(H_0, X, omega, times, options)

.. [1] Ross Shillito, Jonathan A. Gross, Agustin Di Paolo, Élie Genois, and Alexandre Blais. Fast and differentiable simulation of driven quantum systems. *Physical Review Research*, 3(3), September 2021. https://arxiv.org/abs/2012.09282
.. [2] H.P. Breuer and F. Petruccione. *The theory of open quantum systems*. Oxford University Press, Great Clarendon Street, 2002.