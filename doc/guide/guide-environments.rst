.. _environments guide:

************************************
Environments of Open Quantum Systems
************************************

*written by* |pm|_ *and* |gs|_

.. _pm: https://www.menczel.net/
.. |pm| replace:: *Paul Menczel*
.. _gs: https://gsuarezr.github.io/
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

.. note::
    In the literature, the Ohmic spectral density can often be found as :math:`J(\omega) = \alpha \frac{\omega^s}{\omega_c^{s-1}} f(\omega)`,
    where :math:`f(\omega)` with :math:`\lim\limits_{\omega \to \infty} f(\omega) = 0` is known as the cutoff function.
    The cutoff function ensures that the spectral density and its integrals (for example :eq:`cfandps`) do not diverge.
    Sometimes, with sub-Ohmic spectral densities, an infrared cutoff is used as well so that :math:`\lim\limits_{\omega \to 0} J(\omega) = 0`.
    This pre-defined Ohmic environment class is restricted to an exponential cutoff function, which is one of the most commonly used in the literature.
    Other cutoff functions can be used in QuTiP with user-defined environments as explained below.

Substituting the Ohmic spectral density :eq:`ohmicf` into :eq:`cfandps`, the correlation function can be computed analytically: 

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

Multi-exponential approximations to Ohmic environments can be obtained through
the fitting procedures :meth:`approximate<.BosonicEnvironment.approximate>`
The following example shows how to create a sub-Ohmic environment, and how to use
:meth:`approximate<.BosonicEnvironment.approximate>` to fit the real and imaginary parts
of the correlation function with three exponential terms each.

.. plot::
    :context: reset
    :nofigs:

    import numpy as np
    import qutip as qt
    import matplotlib.pyplot as plt

    # Define a sub-Ohmic environment with the given temperature, coupling strength and cutoff
    env = qt.OhmicEnvironment(T=0.1, alpha=1, wc=3, s=0.7)

    # Fit the correlation function with three exponential terms
    tlist = np.linspace(0, 3, 250)
    approx_env, info = env.approximate("cf",tlist, target_rsme=None, Nr_max=3, Ni_max=3, maxfev=1e8)

The environment `approx_env` created here could be used, for example, with the :ref:`HEOM solver<heom>`.
The variable `info` contains info about the convergence of the fit; here, we will just plot the fit together with
the analytical correlation function. Note that a larger number of exponential terms would have yielded a better result.

.. plot::
    :context:

    plt.plot(tlist, np.real(env.correlation_function(tlist)), label='Real part (analytic)')
    plt.plot(tlist, np.real(approx_env.correlation_function(tlist)), '--', label='Real part (fit)')

    plt.plot(tlist, np.imag(env.correlation_function(tlist)), label='Imag part (analytic)')
    plt.plot(tlist, np.imag(approx_env.correlation_function(tlist)), '--', label='Imag part (fit)')

    plt.xlabel('Time')
    plt.ylabel('Correlation function')
    plt.tight_layout()
    plt.legend()


.. _dl env guide:

Drude-Lorentz Environment
^^^^^^^^^^^^^^^^^^^^^^^^^

Drude-Lorentz environments, also known as overdamped environments, can be constructed in QuTiP
using the class :class:`.DrudeLorentzEnvironment`. They are characterized by spectral densities of the form

.. math::
    J(\omega) = \frac{2 \lambda \gamma \omega}{\gamma^{2}+\omega^{2}} ,

where :math:`\lambda` is a coupling strength (with the dimension of energy) and :math:`\gamma` the cutoff frequency.

To compute the corresponding correlation function, one can apply the Matsubara expansion:

.. math::
      C(t) = \sum_{k=0}^{\infty} c_k e^{- \nu_k t}

The coefficients of this expansion are

.. math::

    \nu_{k} = \begin{cases}
        \gamma               & k = 0\\
        {2 \pi k} / {\beta}  & k \geq 1\\
    \end{cases} \;, \qquad
    c_k = \begin{cases}
        \lambda \gamma [\cot(\beta \gamma / 2) - i]                & k = 0\\
        \frac{4 \lambda \gamma \nu_k }{ (\nu_k^2 - \gamma^2)\beta} & k \geq 1\\
    \end{cases} \;.

The function :meth:`approx_by_matsubara<.DrudeLorentzEnvironment.approx_by_matsubara>` creates a multi-exponential
approximation to the Drude-Lorentz environment by truncating this series at a finite index :math:`N_k`.
This approximation can then be used with the HEOM solver, for example.
The :ref:`HEOM section<heom>` of this guide contains further examples using the Drude-Lorentz enviroment.

Similarly, the function :meth:`approx_by_pade<.DrudeLorentzEnvironment.approx_by_pade>` can be used to apply
and truncate the numerically more efficient Padé expansion.

Interestingly, the Drude-Lorentz environment can also be used to generate a Shifted-Drude-Lorentz environment [Kreisbeck12]_,
which can be characterized by spectral densities of the form

.. math::

    J(\omega)=\left[\frac{\gamma \lambda \omega}{\gamma^2+
    \left(\omega+\Omega\right)^2}+\frac{\gamma \lambda \omega}{\gamma^2+
    \left(\omega-\Omega\right)^2}\right]

This can be achieved by summating two Drude-Lorentz Environments with :math:`\gamma \rightarrow \gamma \pm i \Omega`
(where, :math:`\Omega` is the shift) and :math:`\lambda \rightarrow \lambda/2`. The :ref:`HEOM section<heom>` has
an example implementation.

Underdamped Environment
^^^^^^^^^^^^^^^^^^^^^^^

Underdamped environments can be constructed in QuTiP
using the class :class:`.UnderDampedEnvironment`. They are characterized by spectral densities of the form

.. math::
    J(\omega) = \frac{\lambda^{2} \Gamma \omega}{(\omega_0^{2}-
    \omega^{2})^{2}+ \Gamma^{2} \omega^{2}} ,

where :math:`\lambda`, :math:`\Gamma` and :math:`\omega_0` are the coupling strength
(with dimension :math:`(\text{energy})^{3/2}`), the cutoff frequency and the resonance frequency.

Similar to the Drude-Lorentz environment, the correlation function can be approximated by a
Matsubara expansion. This functionality is available with the
:meth:`approx_by_matsubara<.UnderDampedEnvironment.approx_by_matsubara>` function.

For small temperatures, the Matsubara expansion converges slowly. It is recommended to instead use a fitting procedure
for the Matsubara contribution as described in [Lambert19]_.


User-Defined Environments
-------------------------

As stated in the introduction, a bosonic environment is fully characterized
by its temperature and spectral density (SD), or alternatively by its correlation function (CF)
or its power spectrum (PS). QuTiP allows for the creation of an user-defined environment by
specifying either the spectral density, the correlation function, or the power spectrum.

QuTiP then computes the other two functions based on the provided one. To do so, it converts between
the SD and the PS using the formula
:math:`S(\omega) = \operatorname{sign}(\omega)\, J(|\omega|) \bigl[ \coth( \beta\omega / 2 ) + 1 \bigr]`
introduced earlier, and between the PS and the CF using the fast Fourier transform.
The former conversion requires the bath temperature to be specified; the latter requires a cutoff frequency (or cutoff time)
to be provided together with the specified function (SD, CF or PS).
In this way, all characteristic functions can be computed from the specified one.

The following example manually creates an environment with an underdamped spectral density.
It then compares the correlation function obtained via fast Fourier transformation with the Matsubara expansion.
The slow convergence of the Matsubara expansion is visible around :math:`t=0`.

.. plot::
    :context: close-figs

    # Define underdamped environment parameters
    T = 0.1
    lam = 1
    gamma = 2
    w0 = 5

    # User-defined environment based on SD
    def underdamped_sd(w):
        return lam**2 * gamma * w / ((w**2 - w0**2)**2 + (gamma*w)**2)
    env = qt.BosonicEnvironment.from_spectral_density(underdamped_sd, wMax=50, T=T)

    tlist = np.linspace(-2, 2, 250)
    plt.plot(tlist, np.real(env.correlation_function(tlist)), label='FFT')

    # Pre-defined environment and Matsubara approximations
    env2 = qt.UnderDampedEnvironment(T, lam, gamma, w0)
    for Nk in range(0, 11, 2):
        approx_env = env2.approx_by_matsubara(Nk)
        plt.plot(tlist, np.real(approx_env.correlation_function(tlist)), label=f'Nk={Nk}')

    plt.xlabel('Time')
    plt.ylabel('Correlation function (real part)')
    plt.tight_layout()
    plt.legend()


Multi-Exponential Approximations
--------------------------------

Many approaches to simulating the dynamics of an open quantum system strongly coupled to an environment
assume that the environment correlation function can be approximated by a multi-exponential expansion like

.. math::
    C(t) = C_R(t) + \mathrm i C_I(t) , \qquad
    C_{R,I}(t) = \sum_{k=1}^{N_{R,I}} c^{R,I}_k \exp[-\nu^{R,I}_k t]

with small numbers :math:`N_{R,I}` of exponents.
Note that :math:`C_R(t)` and :math:`C_I(t)` are the real and imaginary parts of the correlation function,
but the coefficients :math:`c^{R,I}_k` and exponents :math:`\nu^{R,I}_k` are not required to be real in general.

In the previous sections, various methods of obtaining multi-exponential approximations were introduced.
The methods available in qutip can be roughly put into three categories

- Non-Linear Least Squares:
    - On the Spectral Density (`sd`)
    - On the Correlation Function (`cf`)
    - On the Power Spectrum (`ps`)
- Methods based on the Prony Polynomial
    - Prony on the correlation function(`prony`)
    - The Matrix Pencil method on the correlation function (`mp`) :question:
    - ESPRIT on the correlation function(`esprit`)
- Methods based on rational Approximations
    - The AAA algorithm on the Power Spectrum (`aaa`)
    - ESPIRA-I (`espira-I`) :question:
    - ESPIRA-II (`espira-II`)

.. list-table:: 
   :header-rows: 1
   :widths: auto

   * - Class
     - Requires Extra Information
     - Fast
     - Resilient to Noise
     - Allows Constraints
     - Stable
   * - Non-Linear Least Squares
     - Yes
     - No
     - No
     - Yes
     - No
   * - Prony Polynomial
     - No
     - Yes
     - Partially
     - No
     - Partially
   * - Rational Approximations
     - No
     - Yes
     - Partially
     - Partially
     - Yes

All different approximation schemes are available using the approximate method of
:class:`BosonicEnvironment`,the scheme is chosen by passing the "method" argument,
more information about each approximation scheme can be seen in the tutorials. 

The output of these approximation functions are :class:`.ExponentialBosonicEnvironment` objects.
An :class:`.ExponentialBosonicEnvironment` is basically a collection of :class:`.CFExponent` s, which store (in the bosonic case)
the coefficient, the exponent, and whether the exponent contributes to the real part, the imaginary part, or both.
As we have already seen above, one can then compute the spectral density, correlation function and power spectrum corresponding
to the exponents, in order to compare them to the original, exact environment.

Let :math:`c_k \mathrm e^{-\nu_k t}` be a term in the correlation function (i.e., :math:`c_k = c^R_k` or :math:`c_k = \mathrm i c^I_k`).
The corresponding term in the power spectrum is

.. math::
    S_k(\omega) = 2\Re\Bigr[ \frac{c_k}{\nu_k - \mathrm i\omega} \Bigr]

and, if a temperature has been specified, the corresponding term in the spectral density can be computed as described above.

The following example shows how to manually create an :class:`.ExponentialBosonicEnvironment` for the simple example
:math:`C(t) = c \mathrm e^{-\nu t}` with real :math:`c`, :math:`\nu`. The power spectrum then is a Lorentzian,
:math:`S(\omega) = 2c\nu / (\nu^2 + \omega^2)`.

.. plot::
    :context: close-figs

    c = 1
    nu = 2
    wlist = np.linspace(-3, 3, 250)

    env = qt.ExponentialBosonicEnvironment([c], [nu], [], [])

    plt.figure(figsize=(4, 3))
    plt.plot(wlist, env.power_spectrum(wlist))
    plt.plot(wlist, 2 * c * nu / (nu**2 + wlist**2), '--')
    plt.xlabel('Frequency')
    plt.ylabel('Power spectrum')
    plt.tight_layout()


.. _fermionic environments guide:

Fermionic environments
----------------------

The implementation of fermionic environments in QuTiP is not yet as advanced as the bosonic environments.
Currently, user-defined fermionic environments and fitting are not implemented.

However, the overall structure of fermionic environments in QuTiP is analogous to the bosonic environments.
There is one pre-defined fermionic environment, the Lorentzian environment, and multi-exponential fermionic environments.
Lorentzian environments can be approximated by multi-exponential Matsubara or Padé expansions.

In the fermionic case, we consider the number-conserving Hamiltonian

.. math::
    H = H_s + (B^\dagger c + c^\dagger B) + \sum_k \hbar\omega_k\, b^\dagger_k b_k , \qquad
    B = \sum_k f_k b_k ,

where the bath operators :math:`b_k` and the system operator :math:`c` obey fermionic anti-commutation relations.
In analogy to the bosonic case, we define the spectral density

.. math::
    J(\omega) = 2\pi \sum_k f_k^2\, \delta[\omega - \omega_k] ,

which may however now be defined for all (including negative) frequencies, since the spectrum of each mode is bounded.

The fermionic environment is characterized either by its spectral density, inverse temperature :math:`\beta` and chemical potential :math:`\mu`,
or equivalently by two correlation functions or by two power spectra. The correlation functions are

.. math::
    C^\sigma(t) = \langle B^\sigma(t) B^{-\sigma} \rangle
    = \int_{-\infty}^\infty \frac{\mathrm d\omega}{2\pi}\, J(\omega)\, 
        \mathrm e^{\sigma \mathrm i\omega t}\, f_F(\sigma \beta[\omega - \mu]) ,

where :math:`\sigma = \pm 1`, :math:`B^+ = B^\dagger` and :math:`B^- = B`.
Further, :math:`f_F(x) = (\mathrm e^x + 1)^{-1}` is the Fermi-Dirac function.
Note that we still have :math:`C^\sigma(-t) = C^\sigma(t)^\ast`.
The power spectra are the Fourier transformed correlation functions,

.. math::
    S^\sigma(\omega) = \int_{-\infty}^\infty \mathrm dt\, C^\sigma(t)\, \mathrm e^{-\sigma \mathrm i\omega t}
        = J(\omega) f_F(\sigma\beta[\omega - \mu]) .

Since :math:`f_F(x) + f_F(-x) = 1`, we have :math:`S^+(\omega) + S^-(\omega) = J(\omega)`.

.. note::
    The relationship between the spectral density and the two power spectra (or the two correlation functions) is not one-to-one.
    A pair of functions :math:`S^\pm(\omega)` is physical if they satisfy the condition

    .. math::
        S^-(\omega) = \mathrm e^{\beta(\omega - \mu)}\, S^+(\omega) .

    For the correlation functions, the condition becomes :math:`C^-(t) = \mathrm e^{-\beta\mu}\, C^+(t - \mathrm i\beta)^\ast`.
    For flexibility, we do not enforce the power spectra / correlation functions to be physical in this sense.

.. _lorentzian env guide:

Lorentzian Environment
^^^^^^^^^^^^^^^^^^^^^^

Fermionic Lorentzian environments are represented by the class :class:`.LorentzianEnvironment`.
They are characterized by spectral densities of the form

.. math::
    J(\omega) = \frac{\gamma W^2}{(\omega - \omega_0)^2 + W^2} ,

where :math:`\gamma` is the coupling strength, :math:`W` the spectral width and :math:`\omega_0` the resonance frequency.
Often, the resonance frequency is taken to be equal to the chemical potential of the environment.

As with the bosonic Drude-Lorentz environments, multi-exponential approximations of the correlation functions,

.. math::
    C^\sigma(t) \approx \sum_{k=0}^{N_k} c^\sigma_k e^{- \nu^\sigma_k t} ,

can be obtained using the Matsubara or Padé expansions.
The functions :meth:`approx_by_matsubara<.LorentzianEnvironment.approx_by_matsubara>` and
:meth:`approx_by_pade<.LorentzianEnvironment.approx_by_pade>` implement these approximations in QuTiP,
yielding approximated environments that can be used, for example, with the HEOM solver.
Note that for this type of environment, the Matsubara expansion is very inefficient, converging much more slowly than the Padé expansion.
Typically, at least :math:`N_k \geq 20` is required for good convergence.

For reference, we tabulate the values of the coefficients and exponents in the following.
For the Matsubara expansion, they are

.. math::

    \nu^\sigma_{k} = \begin{cases}
        W - \mathrm i \sigma\, \omega_0                     & k = 0\\
        \frac{(2k - 1) \pi}{\beta} - \mathrm i \sigma\, \mu & k \geq 1\\
    \end{cases} \;, \qquad
    c^\sigma_k = \begin{cases}
        \frac{\gamma W}{2} f_F[\sigma\beta(\omega_0 - \mu) + \mathrm i\, \beta W]    & k = 0\\
        \frac{\mathrm i \gamma W^2}{\beta} \frac{1}{[\mathrm i \sigma (\omega_0 - \mu) + (2k - 1) \pi / \beta]^2 - W^2} & k \geq 1\\
    \end{cases} \;.

The Padé decomposition approximates the Fermi distribution as:

.. math::

    f_F(x) \approx f_F^{\mathrm{approx}}(x) = \frac{1}{2} - \sum_{k=0}^{N_k} \frac{2\kappa_k x}{x^2 + \epsilon_k^2}

where :math:`\kappa_k` and :math:`\epsilon_k` are coefficients that depend on :math:`N_k` and are defined in
`J. Chem Phys 133, "Efficient on the fly calculation of time correlation functions in computer simulations" <https://doi.org/10.1063/1.3491098>`_.
This approach results in the exponents

.. math::

    \nu^\sigma_{k} = \begin{cases}
        W - \mathrm i \sigma\, \omega_0                     & k = 0\\
        \frac{\epsilon_k}{\beta} - \mathrm i \sigma\, \mu    & k \geq 1\\
    \end{cases} \;, \qquad
    c^\sigma_k = \begin{cases}
        \frac{\gamma W}{2} f_F^{\mathrm{approx}}[\sigma\beta(\omega_0 - \mu) + \mathrm i\, \beta W] & k = 0\\
        \frac{\mathrm i\, \kappa_k \gamma W^2}{\beta} \frac{1}{[\mathrm i\sigma(\omega_0 - \mu) + \epsilon_k / \beta]^2 - W^2} & k \geq 1\\
    \end{cases} \;.


Multi-Exponential Fermionic Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analogous to the :class:`.ExponentialBosonicEnvironment` in the bosonic case, the :class:`.ExponentialFermionicEnvironment` describes fermionic environments
where the correlation functions are given by multi-exponential decompositions,

.. math::
    C^\sigma(t) \approx \sum_{k=0}^{N_k^\sigma} c^\sigma_k e^{-\nu^\sigma_k t} .

Like in the bosonic case, the class allows us to automatically compute the spectral density and power spectra that correspond to the
multi-exponential correlation functions.
In this case, they are

.. math::
    S^\sigma(\omega) = \sum_{k=0}^{N_k^\sigma} 2\Re\Bigr[ \frac{c_k^\sigma}{\nu_k^\sigma + \mathrm i \sigma\, \omega} \Bigr]

and :math:`J(\omega) = S^+(\omega) + S^-(\omega)`.


.. plot::
    :context: reset
    :include-source: false
    :nofigs:
