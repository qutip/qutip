.. _environments guide:

************************************
Environments of Open Quantum Systems
************************************

QuTiP can describe environments of open quantum systems.
They can be passed to various solvers, where their influence is taken into account exactly or approximately.
In the following, we will discuss bosonic and fermionic environments.
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

    C(t) = \langle X(t) X \rangle

(where :math:`X(t)` is the interaction picture operator) or its power spectrum

.. math::

    S(\omega) = \int_{-\infty}^\infty \mathrm dt\, C(t)\, \mathrm e^{\mathrm i\omega t} ,

which is the inverse Fourier transform of the correlation function.
The correlation function satisfies the symmetry relation :math:`C(-t) = C(t)^\ast`.

Assuming that the initial state is thermal with inverse temperature :math:`\beta`, the correlation function and the power spectrum can be calculated from the spectral density via

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


Ohmic Environment
-----------------

Todo.


Drude-Lorentz Environment
-------------------------

Todo.


Underdamped Environment
-----------------------

Todo.


User-Defined Environment
------------------------

Todo.


Multi-Exponential Approximations
--------------------------------

Todo.


Fermionic environments
----------------------

Todo.
