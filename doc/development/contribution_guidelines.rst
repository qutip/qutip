.. _development-tests:

*************
Writing tests
*************

Testing guidelines: non-deterministic tests
===========================================

In QuTiP's test there are two common sources of randomness in tests:

- Random inputs:

  The functions tested is itself deterministic, but the inputs are generated randomly.
  It is generally good practice to include some tests with random inputs as these
  can allows to find unexpected edge cases. **Do not fix the seed**, instead
  ensure you also include tests with hand crafted inputs for known common and edge cases.

- Random functions:

  Many functions use random numbers within their internal logic (e.g., the
  Monte-Carlo solver uses random jumps).
  In these cases, most tests should rely
  on the default random output. However, if the function allows for a fixed seed
  as an option, that functionality should also be tested.


To ensure tests are reproducible while maintaining non-deterministic test coverage,
two fixtures are available:

- ``random_generator``: Provides a seeded ``numpy.random.Generator`` instance.
  Use this for functions accepting an explicit RNG or when generating random test inputs.
- ``with_seeded_random``: Calls ``numpy.random.seed(...)`` prior to test execution.
  Use this for legacy code or functions relying on global numpy random state.

The seed used in each case is attached to the test report and will be displayed automatically in the error message if a test fails.

.. code-block:: python

    @pytest.mark.parametrize("number", [
        1,
        # Using np.random in parametrization is acceptable because pytest
        # records the evaluated parameter value in the test failure report.
        np.random.rand(),
    ])
    def test_1(number, random_generator):
        # qutip.rand_* functions accept integers, SeedSequence and
        # pre-made Generator instances for the ``seed`` parameter.
        # We use the fixture pre-made Generator in tests:
        oper = qutip.rand_herm([5, 5], density=number, seed=random_generator)
        assert oper.isherm

    def test_2(with_seeded_random):
        assert np.random.rand() <= 1


**Pre-contribution Check**

  Before submitting a contribution with random tests, please run them several
  hundred times locally to detect intermittent failures. If a failure occurs,
  determine if it is caused by an unsupported edge case or a numerical
  tolerance issue.


- Handling Edge Cases:

  - Logic Errors in the Function:

    If the failure comes from an edge cases that is not, but should be supported,
    please fix it if it is within the scope of your contribution. Otherwise add an
    ``XFAIL`` test and open an GitHub issue for a future fix.

    Example:

      Monte Carlo evolution could detect a "jump" after reaching the ground state
      due to the finite precision of the ODE solver instead of the
      the underlying physics.

  - Input Edge Cases:

    If the issue lies with the input generation, you may need to rethink how
    inputs are created or adjust the options.

    Example:

      An evolution with a random system that do not converge under high coupling.
      The solution would be to scale the random operator to ensure it stays
      within a valid physical range.


Numerical Tolerance Issues:

  Sometimes, either through accumulation of numerical error or the finite precision
  of an iterative method, a test may fail occasionally if the tolerance is too tight.
  In this situation, estimate the distribution of the result over multiple runs
  and set the tolerance to 4-sigma level (failing less than 1 in 10000).

.. note::

  A Note on Floating Point Non-determinism:

  Even if nothing uses random number, some computations may return slightly different
  results across calls. This is usually caused by floating-point arithmetic
  ``(A + B) + C != A + (B + C)`` and asynchronous computation associated with
  parallel maps.  These case should be treated as tolerance issues.
