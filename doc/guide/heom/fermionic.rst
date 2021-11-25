######################
Fermionic Environments
######################

Here we model a single fermion coupled to two electronic leads or reservoirs (e.g.,  this can describe a single quantum dot, a molecular transistor, etc).  Note that in this implementation we primarily follow the definitions used by Christian Schinabeck in his Dissertation https://opus4.kobv.de/opus4-fau/files/10984/DissertationChristianSchinabeck.pdf and related publications.

A notebook containing a complete example similar to this one implemented in
BoFiN can be found in `example notebook 4b <https://github.com/tehruhn/bofin/blob/main/examples/example-4b-fermions-single-impurity-model.ipynb>`__.


Describing the system and bath
------------------------------

First, let us construct the system Hamiltonian, :math:`H_{sys}`, and the initial
system state, ``rho0``:

.. plot::
    :context: reset
    :nofigs:

    from qutip import basis, destroy

    # The system Hamiltonian:
    e1 = 1.  # site energy
    H_sys = e1 * destroy(2).dag() * destroy(2)

    # Initial state of the system:
    rho0 = basis(2,0) * basis(2,0).dag()

Now let us describe the bath properties:

.. plot::
    :context:
    :nofigs:

    # Bath properties:
    Gamma = 0.01   # coupling strength
    W = 1.0  # cut-off
    T = 0.025851991  # temperature
    beta = 1. / T

    # System-bath coupling operator:
    Q = destroy(2)

where :math:`\gamma` (``gamma``), :math:`W` and :math:`T` are the parameters of
an XXX bath, and ``Q`` is the coupling operator between the system and
the bath.

Calculating the bath expansion coefficients
-------------------------------------------

We choose a Lorentzian spectral density for the leads, with a peak at the
chemical potential. The latter simplifies a little the notation required for the
correlation functions, but can be relaxed if neccessary.

.. math::

    J(\omega) = \frac{\Gamma W^2}{((\omega - \mu_K)^2 + W^2)}

Fermi distribution is:

.. math::

    f_F (x) = (\exp(x) + 1)^{-1}

gives correlation functions:

.. math::

    C^{\sigma}_K(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} d\omega e^{\sigma i \omega t} \Gamma_K(\omega) f_F[\sigma\beta(\omega - \mu)]

As with the Bosonic case we can treat these with Matsubara, Pade, or fitting
approaches.

The Pade decomposition approximates the Fermi distubition as:

.. math::

    f_F(x) \approx f_F^{\mathrm{approx}}(x) = \frac{1}{2} - \sum_l^{l_{max}} \frac{2k_l x}{x^2 + \epsilon_l^2}

where :math:`k_l` and :math:`\epsilon_l` are coefficients defined in J. Chem
Phys 133,10106.

Evaluating the integral for the correlation functions gives:

.. math::

    C_K^{\sigma}(t) \approx \sum_{l=0}^{l_{max}} \eta_K^{\sigma_l} e^{-\gamma_{K,\sigma,l}t}

where:

.. math::

    \eta_{K,0} &= \frac{\Gamma_KW_K}{2} f_F^{approx}(i\beta_K W)

    \gamma_{K,\sigma,0} &= W_K - \sigma i\mu_K

    \eta_{K,l \neq 0} &= -i\cdot \frac{k_m}{\beta_K} \cdot \frac{\Gamma_K W_K^2}{-\frac{\epsilon^2_m}{\beta_K^2} + W_K^2}

    \gamma_{K,\sigma,l \neq 0} &= \frac{\epsilon_m}{\beta_K} - \sigma i \mu_K

And now the same numbers calculated in Python:

.. plot::
    :context:
    :nofigs:

    # Imports
    from numpy.linalg import eigvalsh

    # Convenience functions and parameters:
    def deltafun(j,k):
        return 1.0 if j == k else 0.

    lmax = 10  # number of expansion terms to calculate
    theta = 2.0  # bias
    mu_l = theta / 2.
    mu_r = -theta / 2.

    Alpha = np.zeros((2 * lmax, 2 * lmax))
    for j in range(2*lmax):
        for k in range(2*lmax):
            Alpha[j][k] = (
                (deltafun(j, k + 1) + deltafun(j, k - 1))
                / np.sqrt((2 * (j + 1) - 1) * (2 * (k + 1) - 1))
            )

    eigvalsA = eigvalsh(Alpha)

    eps = []
    for val in eigvalsA[0:lmax]:
        eps.append(-2 / val)

    AlphaP = np.zeros((2 * lmax - 1, 2 * lmax - 1))
    for j in range(2 * lmax - 1):
        for k in range(2 * lmax - 1):
            AlphaP[j][k] = (
                (deltafun(j, k + 1) + deltafun(j, k - 1))
                / np.sqrt((2 * (j + 1) + 1) * (2 * (k + 1) + 1))
            )

    eigvalsAP = eigvalsh(AlphaP)

    chi = []
    for val in eigvalsAP[0:lmax - 1]:
        chi.append(-2/val)

    eta_list = [
        0.5 * lmax * (2 * (lmax + 1) - 1) * (
            np.prod([chi[k]**2 - eps[j]**2 for k in range(lmax - 1)]) /
            np.prod([
                eps[k]**2 - eps[j]**2 + deltafun(j, k) for k in range(lmax)
            ])
        )
        for j in range(lmax)
    ]

    kappa = [0] + eta_list
    epsilon = [0] + eps

    def f_approx(x):
        f = 0.5
        for ll in range(1, lmax + 1):
            f = f - 2 * kappa[ll] * x / (x**2 + epsilon[ll]**2)
        return f

    def C(sigma, mu):
        eta_0 = 0.5 * Gamma * W * f_approx(1.0j * beta * W)
        gamma_0 = W - sigma*1.0j*mu
        eta_list = [eta_0]
        gamma_list = [gamma_0]
        if lmax > 0:
            for ll in range(1, lmax + 1):
                eta_list.append(
                    -1.0j * (kappa[ll] / beta) * Gamma * W**2
                    / (-(epsilon[ll]**2 / beta**2) + W**2)
                )
                gamma_list.append(epsilon[ll]/beta - sigma*1.0j*mu)
        return eta_list, gamma_list

    etapL, gampL = C(1.0, mu_l)
    etamL, gammL = C(-1.0, mu_l)

    etapR, gampR = C(1.0, mu_r)
    etamR, gammR = C(-1.0, mu_r)

    ck_plus = etapR + etapL
    vk_plus = gampR + gampL
    ck_minus = etamR + etamL
    vk_minus = gammR + gammL

And finally we are ready to construct the :class:`FermionicBath`:

.. plot::
    :context:
    :nofigs:

    from qutip.nonmarkov.heom import FermionicBath

    # Padé expansion:
    bath = FermionicBath(Q, ck_plus, vk_plus, ck_minus, vk_minus)


System and bath dynamics
------------------------

Now we are ready to construct a solver:

.. plot::
    :context:
    :nofigs:

    from qutip.nonmarkov.heom import HEOMSolver
    from qutip import Options

    max_depth = 2  # maximum hierarchy depth to retain
    options = Options(nsteps=15_000)

    solver = HEOMSolver(H_sys, bath, max_depth=max_depth, options=options)

XXX: Add a note referencing the bosonic description of the returned result.

Below we run the solver again, but use ``e_ops`` to store the expectation
values of the population of the system states and the coherence:

.. plot::
    :context:

    # Define the operators that measure the populations of the two
    # system states:
    P11p = basis(2,0) * basis(2,0).dag()
    P22p = basis(2,1) * basis(2,1).dag()

    # Define the operator that measures the 0, 1 element of density matrix
    # (corresonding to coherence):
    P12p = basis(2,0) * basis(2,1).dag()

    # Run the solver:
    tlist = np.linspace(0, 500, 101)
    result = solver.run(rho0, tlist, e_ops={"11": P11p, "22": P22p, "12": P12p})

    # Plot the results:
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8,8))
    axes.plot(result.times, result.expect["11"], 'b', linewidth=2, label="P11")
    axes.plot(result.times, result.expect["22"], 'r', linewidth=2, label="P22")
    axes.set_xlabel(r't', fontsize=28)
    axes.legend(loc=0, fontsize=12)


Steady state
------------

Using the same solver, we can also determine the steady state of the
combined system and bath using:

.. plot::
    :context:
    :nofigs:

    steady_state, steady_ados = solver.steady_state()


Plotting system currents
------------------------

XXX: Pass in Gamma, W, beta etc as parameters

.. plot::
    :context:
    :nofigs:

    from scipy.integrate import quad

    def analytic_current(theta):
        # Gamma, W, beta, e1
        e1 = destroy(2)

        mu_l = theta / 2.
        mu_r = - theta / 2.

        def f(x):
            return 1 / (np.exp(x) + 1.)

        def Gamma_w(w, mu):
            return Gamma * W**2 / ((w-mu)**2 + W**2)

        def lamshift(w, mu):
            return (w-mu)*Gamma_w(w, mu)/(2*W)

        def integrand(w):
            return (
                ((2 / (np.pi)) * Gamma_w(w, mu_l) * Gamma_w(w, mu_r) *
                    (f(beta * (w - mu_l)) - f(beta * (w - mu_r)))) /
                ((Gamma_w(w, mu_l) + Gamma_w(w, mu_r))**2 + 4 *
                    (w - e1 - lamshift(w, mu_l) - lamshift(w, mu_r))**2)
            )

        def real_func(x):
            return np.real(integrand(x))

        def imag_func(x):
            return np.imag(integrand(x))

        # These integral bounds should be checked to be wide enough if the
        # parameters are changed
        a = -2
        b = 2
        real_integral = quad(real_func, a, b)
        imag_integral = quad(imag_func, a, b)


XXX: make lmax below less invisible

.. plot::
    :context:
    :nofigs:

    def state_current(ado_state):
        level_1_aux = [
            (ado_state.extract(label), ado_state.exps(label)[0])
            for label in ado_state.filter(level=1)
        ]

        def exp_sign(exp):
            return 1 if exp.type == exp.types["+"] else -1

        def exp_op(exp):
            return exp.Q if exp.type == exp.types["+"] else exp.Q.dag()

        # right hand modes are the first k modes in ck/vk_plus and ck/vk_minus
        # and thus the first 2 * k exponents
        k = lmax + 1
        return 1.0j * sum(
            exp_sign(exp) * (exp_op(exp) * aux).tr()
            for aux, exp in level_1_aux[:2 * k]
        )

.. plot::
    :context:
    :nofigs:

    theta_list = np.linspace(-4, 4, 100)
    current_analytical = []
    current_heom = []

    for theta in theta_list:
        ck_plus, vk_plus, ck_minus, vk_minus = XXX
        bath = FermionicBath(Q, ck_plus, vk_plus, ck_minus, vk_minus)
        solver = HEOMSOlver(H_sys, bath, max_depth=2)
        steady_state, steady_ados = solver.steady_state()

        current_analytical.append(analytic_current(theta))
        current_heom.append(state_current(steady_ados))


.. plot::
    :context:

    fig, axes = plt.subplots(figsize=(8, 8))

    axes.plot(theta_list, 2.434e-4 * 1e6 * array(curranalist), color="black", linewidth=3, label= r"Analytical")
    axes.plot(theta_list, -2.434e-4 * 1e6 * array(currPlist), 'r--', linewidth=3, label= r"HEOM $l_{\mathrm{max}}=10$, $n_{\mathrm{max}}=2$")

    axes.locator_params(axis='y', nbins=4)
    axes.locator_params(axis='x', nbins=4)

    axes.set_xticks([-2.5,0.,2.5])
    axes.set_xticklabels([-2.5,0,2.5])

    axes.set_xlabel(r"Bias voltage $\Delta \mu$ ($V$)",fontsize=28)
    axes.set_ylabel(r"Current ($\mu A$)",fontsize=28)
    axes.legend(fontsize=25)


Multiple baths
--------------

As for bosonic baths, the :class:`HEOMSolver` supports having a system interact
with multiple fermionic environments. All that is needed is to supply a list of
baths instead of a singe bath.

.. plot::
    :context: reset
    :include-source: false
    :nofigs:

    # reset the context at the end
