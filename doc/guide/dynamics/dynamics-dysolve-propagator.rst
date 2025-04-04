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

When solving this equation, a time evolution operator :math:`U(t_f, t_i)` (often called a propagator) is introduced and allows a quantum state to evolve from time :math:`t_i` to time :math:`t_f` when applied. To derive an equation for this propagator, a Dyson series (a sum over nested integrals) is used. From this, a general form for the propagator is obtained [2]_ :

.. math::

 \displaystyle U(t_f,t_i) = \sum_{n=0}^{\infty}(-i)^n \int_{t_i}^{t_f} \int_{t_i}^{t_n}... \int_{t_i}^{t_2} H(t_n)H(t_{n-1})...H(t_1) dt_1...dt_{n-1}dt_n

There are many ways to compute such a propagator with code depending on what the system's hamiltonian is.

.. _DysolvePropagator:

Calculating propagators with Dysolve
====================================

For hamiltonians of the form :math:`H(t) = H_0 + \cos(\omega t)X` where a perturbation :math:`X` is added to some quantum system with hamiltonian :math:`H_0`, Dysolve is an excellent method to compute time propagators. Indeed, the calculations can be done in parallel and certain quantities can be reused throughout the whole process. Futhermore, Dysolve can be generalized to more complicated drives, offering a wide range of possible applications. 

After plugging in :math:`H(t) = H_0 + \cos(\omega t)X` in the equation for the propagator and manipulating it, we can find the following result for a time increment :math:`\delta t` :

.. math::

 \displaystyle U(t+\delta t,t) = \sum_{n=0}^{\infty} U^{(n)}(t + \delta t, t)

where 

.. math::

 \displaystyle U^{(n)}(t + \delta t, t) = \sum_{\left\{\boldsymbol{\omega}_n\right\}}e^{i\sum_{p=1}^{n}\boldsymbol{\omega}_n[p]t}S^{(n)}(\boldsymbol{\omega}_n, \delta t)

.. math::

 \displaystyle S^{(n)}(\boldsymbol{\omega}_n, \delta t) = \frac{1}{2^n} \sum_{\boldsymbol{m} \in \mathbb{Z}^{n+1}_+} S^{(n)}_{\boldsymbol{m}}(\boldsymbol{\omega}_n, \delta t)

.. math::

 \displaystyle S^{(n)}_{\boldsymbol{m}}(\boldsymbol{\omega}_n, \delta t) = \int_{0}^{\delta t}\int_{0}^{t_M}...\int_{0}^{t_2} (-iH_0)^{m_n}X(-iH_0)^{m_{n-1}}...X(-iH_0)^{m_0}

.. math::

 \displaystyle \times (-i)^n \prod_{p=1}^{n}e^{i\boldsymbol{\omega}_n[p]t_{l(p)}} dt_1...dt_M

Here, :math:`\{\boldsymbol{\omega}_n\}` is the set of vectors of length :math:`n` of the form :math:`\boldsymbol{\omega}_n = \left[±\omega, ..., ±\omega\right]`, :math:`\boldsymbol{m}= \left[m_n, ..., m_0\right]` where :math:`m_i \in \mathbb{Z}_+`, :math:`M = \left(\sum_{i=0}^n m_i\right) + n` and :math:`l(p) = \left(\sum_{j=0}^{p-1} m_j \right) + p`. Again, some manipulations can be made to rewrite :math:`S^{(n)}(\boldsymbol{\omega}_n, \delta t)`.

.. math::

 \displaystyle S^{(n)}(\boldsymbol{\omega}_n, \delta t) = \frac{-i^n}{2^n} \sum_{k^{(n)}}...\sum_{k^{(0)}} e^{-i\lambda_{k^{(n)}}\delta t}\int_{0}^{\delta t}e^{i(\boldsymbol{\omega}_n[n] + \lambda_{k^{(n)}} - \lambda_{k^{(n-1)}})t_n}dt_n ... \int_{0}^{t_2}e^{i(\boldsymbol{\omega}_n[1] + \lambda_{k^{(1)}} - \lambda_{k^{(0)}})t_1}dt_1

.. math::

 \displaystyle \times \bra{k^{(n)}}X \ket{k^{(n-1)}}...\bra{k^{(1)}}X \ket{k^{(0)}}\ket{k^{(n)}}\bra{k^{(0)}}

This assumes that :math:`H_0` and :math:`X` are written in :math:`H_0`'s basis. The nested integral can be evaluated analytically in code with integration by parts and a recursive function. Also, in practice, the summation over :math:`n` is truncated to the first few terms and the evolution is divided into small time increments. So, with :math:`\tau` being the time ordering operator,

.. math::
 \displaystyle U(T,0) = \tau\prod_{p=0}^{P}U((p+1)\delta t, p\delta t) = \tau\prod_{p=0}^{P}\left(\sum_{n=0}^{\infty}U^{(n)}((p+1)\delta t, p\delta t)\right)

.. math::
 \displaystyle \approx  \tau\prod_{p=0}^{P}\left(\sum_{n=0}^{r}U^{(n)}((p+1)\delta t, p\delta t)\right)

As we can see, computing :math:`S^{(n)}(\boldsymbol{\omega}_n, \delta t)` for a given range on :math:`n` once allows us to calculate as many propagators as we want. Therefore, in the long run and with the right ressources, Dysolve can be an extremely useful and efficient method for calculating propagators of systems with the form described at the beginning of this section.

Zeroth order example
====================
As an example, let's find what :math:`U^{(0)}(t+\delta t, t)` is. First, 

.. math::

 \displaystyle S^{(0)}(\boldsymbol{\omega}_0, \delta t) = \sum_{k^{(0)}}e^{-i\lambda_{k^{(0)}}\delta t} \ket{k^{(0)}}\bra{k^{(0)}} = e^{-iH_0\delta t}

Finally, because :math:`\left\{\boldsymbol{\omega}_0\right\}` is empty, we get

.. math::

 \displaystyle U^{(0)}(t+\delta t, t) = e^{-iH_0\delta t}

.. _dysolve_code_example:

Code example
============

The following code shows a simple example on how to use QuTiP's implementation of Dysolve. Here, :math:`H(t) = \sigma_z + \cos(t)\sigma_x`.

.. code-block:: Python
    
    from qutip.solver.dysolve_propagator import DysolvePropagator, dysolve_propagator
    from qutip import sigmax, sigmaz

    max_order = 5
    omega = 1
    t_i = 0
    t_f = 1
    dt = 0.1
    H_0 = sigmaz()
    X = sigmax()

    #Initialize and call to compute the propagators for each time increment t => U(t + dt, t)
    dysolve = DysolvePropagator(
                max_order, H_0, X, omega
            )
    dysolve(t_i, t_f, dt)
    Us = dysolve.Us

    #Another option is to use the function to get propagators from t_i to each time increment t => U(t, t_i)
    dysolve, propagators = dysolve_propagator(
        max_order, H_0, X, omega, t_i, t_f, dt
    )

.. [1] Ross Shillito, Jonathan A. Gross, Agustin Di Paolo, Élie Genois, and Alexandre Blais. Fast and differentiable simulation of driven quantum systems. *Physical Review Research*, 3(3), September 2021. https://arxiv.org/abs/2012.09282
.. [2] H.P. Breuer and F. Petruccione. *The theory of open quantum systems*. Oxford University Press, Great Clarendon Street, 2002.