.. _fermionic environments guide:

Fermionic Environments
----------------------

Theory
~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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