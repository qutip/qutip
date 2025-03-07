.. _dysolve:

**********************************************
Propagators using Dysolve
**********************************************

Time evolution of a quantum system
==================================
It is common knowledge that the time evolution of a quantum system is given by the Schrödinger equation (:math:`\hbar = 1`) :

.. math::

     \displaystyle i \frac{d}{dt}|\psi(t)> = H(t)|\psi(t)>

When solving this equation, a time evolution operator :math:`U(t_f, t_i)` (often called a propagator) is introduced and allows a quantum state to evolve from time :math:`t_i` to time :math:`t_f` when applied. Its general form is :

.. math::

     \displaystyle U(t_f,t_i) = \sum_{n=0}^{\infty}(-i)^n \int_{t_i}^{t_f} \int_{t_i}^{t_n}... \int_{t_i}^{t_2} H(t_n)H(t_{n-1})...H(t_1) dt_1...dt_{n-1}dt_n

.. _DysolvePropagator:

Dysolve
=======

Dysolve is a method that computes such a propagator for hamiltonians of the form :math:`H(t) = H_0 + \cos(\omega t)X`, often seen in qubit control where a drive changes the state of a qubit (see https://arxiv.org/abs/2012.09282 for the original paper). The calculations can be done in parallel and generalized to a more complicated drive. After plugging the hamiltonian's form in the equation for the propagator and manipulating it, we can find the following result for a time increment :math:`\delta t` :

.. math::

     \displaystyle U(t+\delta t,t) = \sum_{n=0}^{\infty} U^{(n)}(t + \delta t, t)

where 

.. math::

     \displaystyle U^{(n)}(t + \delta t, t) = \sum_{\left\{\boldsymbol{\omega}_n\right\}}e^{i\sum_{p=1}^{n}\boldsymbol{\omega}_n[p]t}S^{(n)}(\boldsymbol{\omega}_n, \delta t)

.. math::

     \displaystyle S^{(n)}(\boldsymbol{\omega}_n, \delta t) = \frac{1}{2^n} \sum_{\boldsymbol{m} \in \mathbb{Z}^{n+1}_+} S^{(n)}_{\boldsymbol{m}}(\boldsymbol{\omega}_n, \delta t)

.. math::

     \displaystyle S^{(n)}_{\boldsymbol{m}}(\boldsymbol{\omega}_n, \delta t) = \int_{0}^{\delta t}\int_{0}^{t_M}...\int_{0}^{t_2} (-iH_0)^{m_n}X...X(-iH_0)^{m_0} \cdot (-i)^n \prod_{p=1}^{n}e^{i\boldsymbol{\omega}_n[p]t_{l(p)}} dt_1 ... dt_M

Here, :math:`\{\boldsymbol{\omega}_n\}` is the set of vectors of length :math:`n` of the form :math:`\boldsymbol{\omega}_n = \left[±\omega, ..., ±\omega\right]` (so :math:`2^n` combinations), :math:`\boldsymbol{m}= \left[m_n, ..., m_0\right]` where :math:`m_i \in \mathbb{Z}_+`, :math:`M = \left(\sum_{i=0}^n m_i\right) + n` and :math:`\left(\sum_{j=0}^{p-1}\right) + p`. Of course, in practice, the summation over :math:`n` is truncated to first few terms and the evolution is divided into small time increments. So,

.. math::
     \displaystyle U(T,0) = \mathcal{T}\prod_{p=0}^{P}U((p+1)\delta t, p\delta t) = \mathcal{T}\prod_{p=0}^{P}\left(\sum_{n=0}^{\infty}U^{(n)}((p+1)\delta t, p\delta t)\right) \approx  \mathcal{T}\prod_{p=0}^{P}\left(\sum_{n=0}^{r}U^{(n)}((p+1)\delta t, p\delta t)\right)

As we can see, computing :math:`S^{(n)}(\boldsymbol{\omega}_n, \delta t)` for a given range on :math:`n` once allows us to calculate as many propagtors as we want. With the right ressources, Dysolve can be an extremely useful tool for calculating propagators.

.. _implementation:

QuTiP implementation
====================
...
