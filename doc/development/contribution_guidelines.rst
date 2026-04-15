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
  can allows to find unexpected edge cases. **Do no fix the seed**, instead
  ensure you also include tests with hand crafted inputs for known common and edge cases.

- Random functions:

  Many functions use random numbers within their internal logic (e.g., the
  Monte-Carlo solver uses random jumps).
  In these cases, most tests should rely
  on the default random output. However, if the function allows for a fixed seed
  as an option, that functionality should also be tested.

**Pre-contribution Check**

  Before submitting a contribution with random tests, please run them several
  hundred times locally to detect intermittent failures. If a failure occurs,
  determine if it is caused by an unsupported edge case or a numerical
  tolerance issue.


- Handling Edge Cases:

  - Logic Errors in the Function:

    If the failure comes from an edge cases that is not, but should be supported,
    please fix it if it is within the scope of your contribution. Otherwise add an
    ``XFAIL`` test as and open an GitHub issue for a future fix.

    Example:

      Monte Carlo evolution could detect a "jump" after reaching the ground state
      due to the finite precision of the ODE solver instead of the rather than
      the underlying physics.

  - Input Edge Cases:

    If the issue lies with the input generation, you may need to rethink how
    inputs are created or adjust the options.

    Example:

      An evolution with a random system that do not converge under high coupling.
      The solution would be to scale the random operator to ensure it stays
      within a valid physical range.


Numerical Tolerance Issues:

  Sometime, either through accumulation of numerical error or finite precision
  of iterative method, a test may fail occasionally if the tolerance is too tight.
  In this situation estimate the distribution of the result over multiple runs
  and set the tolerance to 4-sigma level (failing less than 1 in 10000).

.. note::

  A Note on Floating Point Non-determinism:

  Even if nothing use random number, some computation can return slightly different
  results with across calls. This is usually caused by floating-point arithmetic
  ``(A + B) + C != A + (B + C)`` and asynchronous computation associated with
  parallel maps.  These should generally be treated as tolerance issues.
