.. _environments guide:

************************************
Environments of Open Quantum Systems
************************************

*written by* |pm|_ *and* |gs|_

.. _pm: https://www.menczel.net/
.. |pm| replace:: *Paul Menczel*
.. _gs: https://scholar.google.com/citations?user=yi6jJAQAAAAJ&hl=es
.. |gs| replace:: *Gerardo Suarez*
.. (this is a workaround for italic links in rst)

QuTiP can describe environments of open quantum systems.
They can be passed to various solvers, where their influence is taken into account exactly or approximately.
In the following, we will discuss bosonic and fermionic thermal environments.
In our definitions, we follow [BoFiN23]_.

Note that currently, we only support a single coupling term per environment.
If a more generalized coupling would be useful to you, please let us know on GitHub.


.. _bosonic environments guide:

Bosonic Environments
--------------------

A bosonic environment is described by a continuum of harmonic oscillators.
The open quantum system and its environment are thus described by the Hamiltonian

.. math::

    H = H_{\rm s} + Q \otimes X + \sum_k \hbar\omega_k\, a_k^\dagger a_k , \qquad X = \sum_k g_k (a_k + a_k^\dagger)

(in the case of a single bosonic environment).
Here, :math:`H_{\rm s}` is the Hamiltonian of the open system and :math:`Q` a system coupling operator.
The sums enumerate the environment modes with frequencies :math:`\omega_k` and couplings :math:`g_k`.
The couplings are described by the spectral density

.. math::

    J(\omega) = \pi \sum_k g_k^2 \delta(\omega - \omega_k) .

Equivalently, a bosonic environment can be described by its auto-correlation function

.. math::

    C(t) = \langle X(t) X \rangle = \sum_{k} g_{k}^{2} \Big( \cos(\omega_{k} t)
     \underbrace{( 2 n_{k}+1)}_{\coth(\frac{\beta \omega_{k}}{2})} 
     - i \sin(\omega_{k} t) \Big)

(where :math:`X(t)` is the interaction picture operator) or its power spectrum

.. math::

    S(\omega) = \int_{-\infty}^\infty \mathrm dt\, C(t)\, \mathrm e^{\mathrm i\omega t} ,

which is the inverse Fourier transform of the correlation function.
The correlation function satisfies the symmetry relation :math:`C(-t) = C(t)^\ast`.

Assuming that the initial state is thermal with inverse temperature 
:math:`\beta`,  in the continuum limit the correlation function and the power 
spectrum can be calculated from the spectral density via

.. math::
    :label: cfandps

    \begin{aligned}
        C(t) &= \int_0^\infty \frac{\mathrm d\omega}{\pi}\, J(\omega) \Bigl[ \coth\Bigl( \frac{\beta\omega}{2} \Bigr) \cos\bigl( \omega t \bigr) - \mathrm i \sin\bigl( \omega t \bigr) \Bigr] , \\
        S(\omega) &= \operatorname{sign}(\omega)\, J(|\omega|) \Bigl[ \coth\Bigl( \frac{\beta\omega}{2} \Bigr) + 1 \Bigr] .
    \end{aligned}

Here, :math:`\operatorname{sign}(\omega) = \pm 1` depending on the sign of :math:`\omega`.
At zero temperature, these equations become :math:`C(t) = \int_0^\infty \frac{\mathrm d\omega}{\pi} J(\omega) \mathrm e^{-\mathrm i\omega t}` and :math:`S(\omega) = 2 \Theta(\omega) J(|\omega|)`, where :math:`\Theta` is the Heaviside function.

If the environment is coupled weakly to the open system, the environment induces quantum jumps with transition rates :math:`\gamma \propto S(-\Delta\omega)`, where :math:`\Delta\omega` is the energy change in the system corresponding to the quantum jump.
In the strong coupling case, QuTiP provides exact integration methods based on multi-exponential decompositions of the correlation function, see below.

.. note::
    We generally assume that the frequencies :math:`\omega_k` are all positive and hence :math:`J(\omega) = 0` for :math:`\omega \leq 0`.
    To handle a spectral density :math:`J(\omega)` with support at negative frequencies, one can use an effective spectral density :math:`J'(\omega) = J(\omega) - J(-\omega)`.
    This process might result in the desired correlation function, because

    .. math::

        \int_0^\infty \frac{\mathrm d\omega}{\pi}\, J'(\omega) \bigl[ \cdots \bigr] = \int_{-\infty}^\infty \frac{\mathrm d\omega}{\pi}\, J(\omega) \bigl[ \cdots \bigr] ,

    where :math:`[\cdots]` stands for the square brackets in Eq. :eq:`cfandps`.
    Note however that the derivation of Eq. :eq:`cfandps` is not valid in this situation, since the environment does not have a thermal state.

.. note::
    Note that the expressions provided above for :math:`S(\omega)` are ill-defined at :math:`\omega=0`.
    For zero temperature, we simply have :math:`S(0) = 0`.

    For non-zero temperature, one obtains

    .. math::

        S(0) = 2\beta^{-1}\, J'(0) .

    Hence, :math:`S(0)` diverges if the spectral density is sub-ohmic.


Pre-defined Environments
------------------------

Ohmic Environment
^^^^^^^^^^^^^^^^^

Ohmic environments can be constructed in QuTiP using the class :class:`.OhmicEnvironment`.
They are characterized by spectral densities of the form

.. math::
    :label: ohmicf

    J(\omega) = \alpha \frac{\omega^s}{\omega_c^{s-1}} e^{-\omega / \omega_c} ,

where :math:`\alpha` is a dimensionless parameter that indicates the coupling strength,
:math:`\omega_{c}` is the cutoff frequency, and :math:`s` is a parameter that determines the low-frequency behaviour.
Ohmic environments are usually classified according to this parameter as

* Sub-Ohmic (:math:`s<1`)
* Ohmic (:math:`s=1`)
* Super-Ohmic (:math:`s>1`).

Substituting this spectral density into :eq:`cfandps`, the correlation function can be computed analytically: 

.. math::
    C(t)= \frac{\alpha}{\pi} w_{c}^{1-s} \beta^{-(s+1)} \Gamma(s+1)
    \left[ \zeta\left(s+1,\frac{1+\beta w_{c} -i w_{c} t}{\beta w_{c}}
    \right) +\zeta\left(s+1,\frac{1+ i w_{c} t}{\beta w_{c}}\right)
    \right] ,

where :math:`\beta` is the inverse temperature, :math:`\Gamma` the Gamma function, and :math:`\zeta` the Hurwitz zeta function.
The zero temperature case can be obtained by taking the limit :math:`\beta \to \infty`, which results in 

.. math::
    C(t) = \frac{\alpha}{\pi} \omega_c^2\, \Gamma(s+1) (1+ i \omega_{c} t)^{-(s+1)} .

The evaluation of the zeta function for complex arguments requires `mpmath`, so certain features of the Ohmic enviroment are 
only available if `mpmath` is installed.

Multi-exponential approximations to Ohmic environments can currently only obtained through
the fitting procedures :meth:`approx_by_cf_fit<.BosonicEnvironment.approx_by_cf_fit>`
and :meth:`approx_by_sd_fit<.BosonicEnvironment.approx_by_sd_fit>`.

.. note::
    In the literature, the Ohmic spectral density can often be found as :math:`J(\omega) = \alpha \frac{\omega^s}{\omega_c^{s-1}} f(\omega)`,
    where :math:`f(\omega)` with :math:`\lim\limits_{\omega \to \infty} f(\omega) = 0` is known as the cutoff function.
    The cutoff function ensures that the spectral density and its integrals (for example :eq:`cfandps`) do not diverge.
    Sometimes, with sub-Ohmic spectral densities, an infrared cutoff is used as well so that :math:`\lim\limits_{\omega \to 0} J(\omega) = 0`.
    This pre-defined Ohmic environment class is restricted to an exponential cutoff function, which is one of the most commonly used in the literature.
    Other cutoff functions can be used in QuTiP with user-defined environments as explained below.

Drude-Lorentz Environment
^^^^^^^^^^^^^^^^^^^^^^^^^

Drude-Lorentz environments, also known as overdamped environments, can be constructed in QuTiP
using the class :class:`.DrudeLorentzEnvironment`. They are characterized by spectral densities of the form

.. math::
    J(\omega) = \frac{2 \lambda \gamma \omega}{\gamma^{2}+\omega^{2}} ,

where 



Underdamped Environment
^^^^^^^^^^^^^^^^^^^^^^^

An Underdamped enviroment is characterized by the spectral density 

.. math::
    J(\omega) = \frac{\lambda^{2} \Gamma \omega}{(\omega_{c}^{2}-
    \omega^{2})^{2}+ \Gamma^{2} \omega^{2}}




User-Defined Environments
-------------------------

As stated in the introduction a thermal Bosonic environment is fully characterized
by its temperature and spectral density, or alternatively by its correlation function
or power spectrum. QuTiP Allows for the creation of an User defined environment by
specifying either

* The Spectral Density 
* The Correlation function
* The  Power spectrum

While temperature is an optional parameter, it is needed to fully characterize
the environment. If it is not provided then one cannot recover the unspecified 
functions 

TODO: Very Clear example.



Multi-Exponential Approximations
--------------------------------

TODO.


Fermionic environments
----------------------

Todo.
