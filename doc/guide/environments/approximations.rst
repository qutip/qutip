.. _environment approximations guide:

Approximations
--------------

.. admonition:: Fermionic Environments

    Approximation methods for fermionic environments are still under development.
    Currently, the only available fermionic approximation methods are
    :meth:`approx_by_matsubara<.LorentzianEnvironment.approx_by_matsubara>` and
    :meth:`approx_by_pade<.LorentzianEnvironment.approx_by_pade>` for Lorentzian environments.
    In a future release, we plan to add fitting methods for user-defined fermionic environments.
    The aforementioned methods will then be replaced by a single ``approximate`` method as in the bosonic case.

Approximating Bosonic Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

QuTiP contains various methods for approximating the correlation functions of environments with multi-exponential functions.
All of these methods are available through the :meth:`approximate<.BosonicEnvironment.approximate>`
function of :class:`BosonicEnvironment` objects. The method to use is chosen by passing the ``method`` argument.

Some of the approximation methods are based on analytical expressions, while others are based on numerical fitting.
The following table shows which methods are available for which type of environment.
Below, we then explain the fitting methods in more detail,
and we document the parameters for the :meth:`approximate<.BosonicEnvironment.approximate>` function.

TODO: table

..
    Rows: Ohmic, DL, UD, User-defined
    Columns: Matsubara, Pade, Fitting

Fitting Methods
~~~~~~~~~~~~~~~

The fitting methods available in QuTiP can be roughly put into three categories:

- Non-Linear least squares:
    - On the Correlation Function (``"cf"``)
    - On the Power Spectrum (``"ps"``)
    - On the Spectral Density (``"sd"``)
- Methods based on the Prony polynomial
    - Prony on the correlation function (``"prony"``)
    - The Matrix Pencil method on the correlation function (``"mp"``)
    - ESPRIT on the correlation function(``"esprit"``)
- Methods based on rational approximations
    - The AAA algorithm on the Power Spectrum (``"aaa"``)
    - ESPIRA-I (``"espira-I"``)
    - ESPIRA-II (``"espira-II"``)

The following table highlights the strengths and weaknesses of each fitting method.
TODO: explain a bit the meaning of the columns?

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

.. _environment approximations api:

API Documentation
~~~~~~~~~~~~~~~~~

.. contents:: Full List of Approximation Methods:
  :local:

..
    Note: the formatting of the docstrings in the rst here is slightly different
    from the one in the codebase. The formatting here is like the rst output of
    numpydoc, which can be viewed with this trick: https://stackoverflow.com/a/31648880

``"matsubara"`` | ``"pade"`` Analytical Expansions
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

.. method:: approximate("matsubara" | "pade", Nk: int, combine: bool = True, compute_delta: Literal[False] = False, tag: Any = None) -> ExponentialBosonicEnvironment
    :no-index:

.. method:: approximate("matsubara" | "pade", Nk: int, combine: bool = True, compute_delta: Literal[True] = True, tag: Any = None) -> tuple[ExponentialBosonicEnvironment, float]
    :no-index:

    Generates an approximation to the environment by truncating its
    Matsubara or Pade expansion.

    :Parameters:

        **Nk** : int
            Number of terms to include. For a Drude-Lorentz environment
            (underdamped environment), the real part of the correlation function
            will include `Nk+1` (`Nk+2`) terms and the imaginary part `1` term
            (`2` terms).

        **combine** : bool, default `True`
            Whether to combine exponents with the same frequency.

        **compute_delta** : bool, default `False`
            Whether to compute and return the approximation discrepancy
            (see below).

        **tag** : optional, str, tuple or any other object
            An identifier (name) for the approximated environment. If not
            provided, a tag will be generated from the tag of this environment.

    :Returns:

        **approx_env** : :class:`.ExponentialBosonicEnvironment`
            The approximated environment with multi-exponential correlation
            function.

        **delta** : float, optional
            The approximation discrepancy. That is, the difference between the
            true correlation function of the Drude-Lorentz environment and the
            sum of the ``Nk`` exponential terms is approximately ``2 * delta *
            dirac(t)``, where ``dirac(t)`` denotes the Dirac delta function.
            It can be used to create a "terminator" term to add to the system
            dynamics to take this discrepancy into account, see
            :func:`.system_terminator`.

``"cf"`` Fit Correlation Function with Exponentials
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

.. method:: approximate("cf", tlist: ArrayLike, target_rsme: float = 2e-5, Nr_max: int = 10, Ni_max: int = 10, guess: list[float] = None, lower: list[float] = None, upper: list[float] = None, sigma: float | ArrayLike = None, maxfev: int = None, full_ansatz: bool = False, combine: bool = True, tag: Any = None) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]
    :no-index:

    Generates an approximation to the environment by fitting its
    correlation function with a multi-exponential ansatz. The number of
    exponents is determined iteratively based on reducing the normalized
    root mean squared error below a given threshold.

    Specifically, the real and imaginary parts are fit by the following
    model functions:

    .. math::
        \operatorname{Re}[C(t)] = \sum_{k=1}^{N_r} \operatorname{Re}\Bigl[
            (a_k + \mathrm i d_k) \mathrm e^{(b_k + \mathrm i c_k) t}\Bigl]
            ,
        \\
        \operatorname{Im}[C(t)] = \sum_{k=1}^{N_i} \operatorname{Im}\Bigl[
            (a'_k + \mathrm i d'_k) \mathrm e^{(b'_k + \mathrm i c'_k) t}
            \Bigr].

    If the parameter `full_ansatz` is `False`, :math:`d_k` and :math:`d'_k`
    are set to zero and the model functions simplify to

    .. math::
        \operatorname{Re}[C(t)] = \sum_{k=1}^{N_r}
            a_k  e^{b_k  t} \cos(c_{k} t)
            ,
        \\
        \operatorname{Im}[C(t)] = \sum_{k=1}^{N_i}
            a'_k  e^{b'_k  t} \sin(c'_{k} t) .

    The simplified version offers faster fits, however it fails for
    anomalous spectral densities with
    :math:`\operatorname{Im}[C(0)] \neq 0` as :math:`\sin(0) = 0`.

    :Parameters:

        **tlist** : array_like
            The time range on which to perform the fit.

        **target_rmse** : optional, float
            Desired normalized root mean squared error (default `2e-5`). Can be
            set to `None` to perform only one fit using the maximum number of
            modes (`Nr_max`, `Ni_max`).

        **Nr_max** : optional, int
            The maximum number of modes to use for the fit of the real part
            (default 10).

        **Ni_max** : optional, int
            The maximum number of modes to use for the fit of the imaginary
            part (default 10).

        **guess** : optional, list of float
            Initial guesses for the parameters :math:`a_k`, :math:`b_k`, etc.
            The same initial guesses are used for all values of k, and for
            the real and imaginary parts. If `full_ansatz` is True, `guess` is
            a list of size 4, otherwise, it is a list of size 3.
            If none of `guess`, `lower` and `upper` are provided, these
            parameters will be chosen automatically.

        **lower** : optional, list of float
            Lower bounds for the parameters :math:`a_k`, :math:`b_k`, etc.
            The same lower bounds are used for all values of k, and for
            the real and imaginary parts. If `full_ansatz` is True, `lower` is
            a list of size 4, otherwise, it is a list of size 3.
            If none of `guess`, `lower` and `upper` are provided, these
            parameters will be chosen automatically.

        **upper** : optional, list of float
            Upper bounds for the parameters :math:`a_k`, :math:`b_k`, etc.
            The same upper bounds are used for all values of k, and for
            the real and imaginary parts. If `full_ansatz` is True, `upper` is
            a list of size 4, otherwise, it is a list of size 3.
            If none of `guess`, `lower` and `upper` are provided, these
            parameters will be chosen automatically.

        **sigma** : optional, float or list of float
            Adds an uncertainty to the correlation function of the environment,
            i.e., adds a leeway to the fit. This parameter is useful to adjust
            if the correlation function is very small in parts of the time
            range. For more details, see the documentation of
            ``scipy.optimize.curve_fit``.

        **maxfev** : optional, int
            Number of times the parameters of the fit are allowed to vary
            during the optimization (per fit).

        **full_ansatz** : optional, bool (default False)
            If this is set to False, the parameters :math:`d_k` are all set to
            zero. The full ansatz, including :math:`d_k`, usually leads to
            significantly slower fits, and some manual tuning of the `guesses`,
            `lower` and `upper` is usually needed. On the other hand, the full
            ansatz can lead to better fits with fewer exponents, especially
            for anomalous spectral densities with
            :math:`\operatorname{Im}[C(0)] \neq 0` for which the simplified
            ansatz will always give :math:`\operatorname{Im}[C(0)] = 0`.
            When using the full ansatz with default values for the guesses and
            bounds, if the fit takes too long, we recommend choosing guesses
            and bounds manually.

        **combine** : optional, bool (default True)
            Whether to combine exponents with the same frequency. See
            :meth:`combine <.ExponentialBosonicEnvironment.combine>` for
            details.

        **tag** : optional, str, tuple or any other object
            An identifier (name) for the approximated environment. If not
            provided, a tag will be generated from the tag of this environment.

    :Returns:

        **approx_env** : :class:`.ExponentialBosonicEnvironment`
            The approximated environment with multi-exponential correlation
            function.

        **fit_info** : dictionary
            A dictionary containing the following information about the fit.

            "Nr"
                The number of terms used to fit the real part of the
                correlation function.
            "Ni"
                The number of terms used to fit the imaginary part of the
                correlation function.
            "fit_time_real"
                The time the fit of the real part of the correlation function
                took in seconds.
            "fit_time_imag"
                The time the fit of the imaginary part of the correlation
                function took in seconds.
            "rmse_real"
                Normalized mean squared error obtained in the fit of the real
                part of the correlation function.
            "rmse_imag"
                Normalized mean squared error obtained in the fit of the
                imaginary part of the correlation function.
            "params_real"
                The fitted parameters (array of shape Nx3 or Nx4) for the real
                part of the correlation function.
            "params_imag"
                The fitted parameters (array of shape Nx3 or Nx4) for the
                imaginary part of the correlation function.
            "summary"
                A string that summarizes the information about the fit.


``"ps"`` Fit Power Spectrum with Lorentzians
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

.. method:: approximate("ps", wlist: ArrayLike, target_rmse: float = 5e-6, Nmax: int = 5, guess: list[float] = None, lower: list[float] = None, upper: list[float] = None, sigma: float | ArrayLike = None, maxfev: int = None, combine: bool = True, tag: Any = None) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]
    :no-index:

    Generates an approximation to this environment by fitting its power
    spectrum with the Fourier transform of decaying exponentials (i.e., with generalized Lorentzians). The
    number of Lorentzians is determined iteratively based on reducing
    the normalized root mean squared error below a given threshold.

    Specifically, the power spectrum is fit by the following model function:

    .. math::
        S(\omega) = \sum_{k=1}^{N}\frac{2(a_k c_k + b_k (d_k - \omega))}{(\omega - d_k)^2 + c_k^2}

    :Parameters:

        **wlist** : array_like
            The frequency range on which to perform the fit.

        **target_rmse** : optional, float
            Desired normalized root mean squared error (default `5e-6`). Can be
            set to `None` to perform only one fit using the maximum number of
            modes (`Nmax`).

        **Nmax** : optional, int
            The maximum number of Lorentzians to use for the fit (default 5).

        **guess** : optional, list of float
            Initial guesses for the parameters :math:`a_k`, :math:`b_k`,
            :math:`c_k` and :math:`d_k`. The same initial guesses are used for all values of k.
            If none of `guess`, `lower` and `upper` are provided, these
            parameters will be chosen automatically.

        **lower** : optional, list of float
            Lower bounds for the parameters :math:`a_k`, :math:`b_k`,
            :math:`c_k` and :math:`d_k`. The same lower bounds are used for all
            values of k.
            If none of `guess`, `lower` and `upper` are provided, these
            parameters will be chosen automatically.

        **upper** : optional, list of float
            Upper bounds for the parameters :math:`a_k`, :math:`b_k`,
            :math:`c_k` and :math:`d_k`. The same upper bounds are used for all values of k.
            If none of `guess`, `lower` and `upper` are provided, these
            parameters will be chosen automatically.

        **sigma** : optional, float or list of float
            Adds an uncertainty to the power spectrum of the environment,
            i.e., adds a leeway to the fit. This parameter is useful to adjust
            if the power spectrum is very small in parts of the frequency
            range. For more details, see the documentation of
            ``scipy.optimize.curve_fit``.

        **maxfev** : optional, int
            Number of times the parameters of the fit are allowed to vary
            during the optimization (per fit).

        **combine** : optional, bool (default True)
            Whether to combine exponents with the same frequency. See
            :meth:`combine <.ExponentialBosonicEnvironment.combine>` for
            details.

        **tag** : optional, str, tuple or any other object
            An identifier (name) for the approximated environment. If not
            provided, a tag will be generated from the tag of this environment.

    :Returns:

        **approx_env** : :class:`.ExponentialBosonicEnvironment`
            The approximated environment with multi-exponential correlation
            function.

        **fit_info** : dictionary
            A dictionary containing the following information about the fit.

            "N"
                The number of underdamped terms used in the fit.
            "fit_time"
                The time the fit took in seconds.
            "rmse"
                Normalized mean squared error obtained in the fit.
            "params"
                The fitted parameters (array of shape Nx4).
            "summary"
                A string that summarizes the information about the fit.


``"sd"`` Fit Spectral Density with Underdamped SDs
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

.. method:: approximate("sd", wlist: ArrayLike, Nk: int = 1, target_rmse: float = 5e-6, Nmax: int = 10, guess: list[float] = None, lower: list[float] = None, upper: list[float] = None, sigma: float | ArrayLike = None, maxfev: int = None, combine: bool = True, tag: Any = None) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]
    :no-index:

    Generates an approximation to the environment by fitting its spectral
    density with a sum of underdamped terms. Each underdamped term
    effectively acts like an underdamped environment. We use the known
    exponential decomposition of the underdamped environment, keeping `Nk`
    Matsubara terms for each. The number of underdamped terms is determined
    iteratively based on reducing the normalized root mean squared error
    below a given threshold.

    Specifically, the spectral density is fit by the following model
    function:

    .. math::
        J(\omega) = \sum_{k=1}^{N} \frac{2 a_k b_k \omega}{\left(\left(
            \omega + c_k \right)^2 + b_k^2 \right) \left(\left(
            \omega - c_k \right)^2 + b_k^2 \right)}

    :Parameters:

        **wlist** : array_like
            The frequency range on which to perform the fit.

        **Nk** : optional, int
            The number of Matsubara terms to keep in each mode (default 1).

        **target_rmse** : optional, float
            Desired normalized root mean squared error (default `5e-6`). Can be
            set to `None` to perform only one fit using the maximum number of
            modes (`Nmax`).

        **Nmax** : optional, int
            The maximum number of modes to use for the fit (default 10).

        **guess** : optional, list of float
            Initial guesses for the parameters :math:`a_k`, :math:`b_k` and
            :math:`c_k`. The same initial guesses are used for all values of
            k.
            If none of `guess`, `lower` and `upper` are provided, these
            parameters will be chosen automatically.

        **lower** : optional, list of float
            Lower bounds for the parameters :math:`a_k`, :math:`b_k` and
            :math:`c_k`. The same lower bounds are used for all values of
            k.
            If none of `guess`, `lower` and `upper` are provided, these
            parameters will be chosen automatically.

        **upper** : optional, list of float
            Upper bounds for the parameters :math:`a_k`, :math:`b_k` and
            :math:`c_k`. The same upper bounds are used for all values of
            k.
            If none of `guess`, `lower` and `upper` are provided, these
            parameters will be chosen automatically.

        **sigma** : optional, float or list of float
            Adds an uncertainty to the spectral density of the environment,
            i.e., adds a leeway to the fit. This parameter is useful to adjust
            if the spectral density is very small in parts of the frequency
            range. For more details, see the documentation of
            ``scipy.optimize.curve_fit``.

        **maxfev** : optional, int
            Number of times the parameters of the fit are allowed to vary
            during the optimization (per fit).

        **combine** : optional, bool (default True)
            Whether to combine exponents with the same frequency. See
            :meth:`combine <.ExponentialBosonicEnvironment.combine>` for
            details.

        **tag** : optional, str, tuple or any other object
            An identifier (name) for the approximated environment. If not
            provided, a tag will be generated from the tag of this environment.

    :Returns:

        **approx_env** : :class:`.ExponentialBosonicEnvironment`
            The approximated environment with multi-exponential correlation
            function.

        **fit_info** : dictionary
            A dictionary containing the following information about the fit.

            "N"
                The number of underdamped terms used in the fit.
            "Nk"
                The number of Matsubara modes included per underdamped term.
            "fit_time"
                The time the fit took in seconds.
            "rmse"
                Normalized mean squared error obtained in the fit.
            "params"
                The fitted parameters (array of shape Nx3).
            "summary"
                A string that summarizes the information about the fit.


``"aaa"`` Fit Power Spectrum using AAA Algorithm
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

.. method:: approximate("aaa", wlist: ArrayLike, tol: float = 1e-13, N_max: int = 10, combine: bool = True, tag: Any = None) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]
    :no-index:

    Generates an approximation to the environment by fitting its power
    spectrum using the AAA algorithm. The power spectrum is fit to a rational
    polynomial of the form

    .. math::
        S(\omega) = 2 \Re \left( \sum_{k} \frac{c_{k}}{\nu_{k} - \mathrm i \omega} \right)

    By isolating the poles and residues of a section of the complex plane,
    the correlation function can be reconstructed as a sum of decaying
    exponentials. The main benefit of this method is that it does not
    require much knowledge about the function to be fit. On the downside,
    if many poles are around the origin, it might require the sample points
    to be used for the fit to be a large dense range which makes this
    algorithm consume a lot of RAM (it will also be slow if asking for many
    exponents).

    :Parameters:

        **wlist** : array_like
            The frequency range on which to perform the fit. With this method
            typically logarithmic spacing works best.

        **tol** : optional, float
            Relative tolerance used to stop the algorithm, if an iteration
            contribution is less than the tolerance the fit is stopped (default `1e-13`).

        **Nmax** : optional, int
            The maximum number of exponents desired. Corresponds to the
            maximum number of iterations for the AAA algorithm (default 10).

        **combine** : optional, bool (default True)
            Whether to combine exponents with the same frequency. See
            :meth:`combine <.ExponentialBosonicEnvironment.combine>` for
            details.

        **tag** : optional, str, tuple or any other object
            An identifier (name) for the approximated environment. If not
            provided, a tag will be generated from the tag of this environment.

    :Returns:

        **approx_env** : :class:`.ExponentialBosonicEnvironment`
            The approximated environment with multi-exponential correlation
            function.


``"prony"`` | ``"mp"`` | ``"esprit"`` | ``"espira-I"`` | ``"espira-II"`` Prony-Based and ESPIRA
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

.. method:: approximate("prony" | "mp" | "esprit" | "espira-I" | "espira-II", tlist: ArrayLike, Nr: int = 3, Ni: int = 3, separate: bool = False, combine: bool = True, tag: Any = None) -> ExponentialBosonicEnvironment
    :no-index:

    Generates an approximation to the environment by fitting its
    correlation function using methods based on the Prony polynomial:

    - ``"prony"``  For the Prony method
    - ``"mp"``  For the Matrix Pencil method
    - ``"esprit"``  For the "Estimation of Signal Parameters via Rotational Invariant Techniques" method

    or methods based on the AAA algorithm:

    - ``"espira-I"``  For the "Estimation of Signal Parameters by Iterative Rational Approximation" method
    - ``"espira-II"``  For the modified ESPIRA method based on matrix pencils for Loewner matrices

    :Parameters:

        **tlist** : array_like
            The time range on which to perform the fit.

        **Nr** : optional, int
            The number of exponents desired to describe the imaginary part of
            the correlation function. It defaults to 3

        **Nr** : optional, int
            The number of exponents desired to describe the real part of
            the correlation function. It defaults to 3

        **separate**: optional, bool
            When True real and imaginary parts are fit separately

        **combine** : optional, bool (default True)
            Whether to combine exponents with the same frequency. See
            :meth:`combine <.ExponentialBosonicEnvironment.combine>` for
            details.

        **tag** : optional, str, tuple or any other object
            An identifier (name) for the approximated environment. If not
            provided, a tag will be generated from the tag of this environment.

    :Returns:

        **approx_env** : :class:`.ExponentialBosonicEnvironment`
            The approximated environment with multi-exponential correlation
            function.
