.. QuTiP
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _measurement:

******************************
Measurement of Quantum Objects
******************************

.. testsetup:: [measurement]

   from qutip import basis, sigmax, sigmaz

   from qutip.measurement import measure, measurement_statistics


.. _measurement-intro:

Introduction
============

Measurement is a fundamental part of the standard formulation of quantum
mechanics and is the process by which classical readings are obtained from
a quantum object. Although the intrepretation of the procedure is at times
contentious, the procedure itself is mathematically straightforward and is
described in many good introductory texts.

Here we will show you how to perform simple measurement operations on QuTiP
objects.

.. _measurement-basic:

Performing a basic measurement
------------------------------

First we need to select some states to measure. For now, let us create an *up*
state and a *down* state:

.. testcode:: [measurement]

   up = basis(2, 0)

   down = basis(2, 1)

which represent spin-1/2 particles with their spin pointing either up or down
along the z-axis.

We choose what to measure by selecting a *measurement operator*. For example,
we could select :func:`~qutip.sigmaz` which measures the z-component of the
spin of a spin-1/2 particle, or :func:`~qutip.sigmax` which measures the
x-component:

.. testcode:: [measurement]

   spin_z = sigmaz()

   spin_x = sigmax()

How do we know what these operators measure? The answer lies in the measurement
procedure itself:

* A quantum measurement tranforms the state being measured by projecting it into
  one of the eigenvectors of the measurement operator.

* Which eigenvector to project onto is chosen probabilistically according to the
  square of the amplitude of the state in the direction of the eigenvector.

* The value returned by the measurement is the eigenvalue corresponding to the
  chosen eigenvector.

.. note:: [measurement]

   How to interpret this "random choosing" is the famous
   "quantum measurement problem".

The eigenvectors of `spin_z` are the states with their spin pointing either up
or down, so it measures the component of the spin along the z-axis.

The eigenvectors of `spin_x` are the states with their spin pointing either
left or right, so it measures the component of the spin along the x-axis.

When we measure our `up` and `down` states using the operator `spin_z`, we
always obtain:
.. testdoc:: [measurement]
    :hide:

    np.random.seed(42)

.. testcode:: [measurement]

   measure(up, spin_z) == (1.0, up)

   measure(down, spin_z) == (-1.0, down)

because `up` is the eigenvector of `spin_z` with eigenvalue `1.0` and `down`
is the eigenvector with eigenvalue `-1.0`. The minus signs are just an
arbitrary global phase -- `up` and `-up` represent the same quantum state.

Neither eigenvector has any component in the direction of the other (they are
orthogonal), so `measure(spin_z, up)` returns the state `up` 100% percent of the
time and `measure(spin_z, down)` returns the state `down` 100% of the time.

Note how :func:`~qutip.measurement.measure` returns a pair of values. The
first is the measured value, i.e. an eigenvalue of the operator (e.g. `1.0`),
and the second is the state of the quantum system after the measurement,
i.e. an eigenvector of the operator (e.g. `up`).

Now let us consider what happens if we measure the x-component of the spin
of `up`:

.. testcode:: [measurement]

   measure(up, spin_x)

The `up` state is not an eigenvector of `spin_x`. `spin_x` has two eigenvectors
which we will call `left` and `right`. The `up` state has equal components in
the direction of these two vectors, so measurement will select each of them
50% of the time.

These `left` and `right` states are:

.. testcode:: [measurement]

   left = (up - down).unit()

   right = (up + down).unit()

When `left` is chosen, the result of the measurement will be `(-1.0, -left)`.

When `right` is chosen, the result of measurement with be `(1.0, right)`.

Now you know how to measure quantum states in QuTiP!

The `measure` function can perform measurements on density matrices too. You
can read about these and other details at :func:`~qutip.measurement.measure`.

.. _measurement-statistics:

Obtaining measurement statistics
--------------------------------

You've just learned how to perform measurements in QuTiP, but you've also
learned that measurements are probabilistic. What if instead of just making
a single measurement, we want to determine the probability distribution of
a large number of measurements?

One way would be to repeat the measurement many times -- and this is what
happens in many quantum experiments. In QuTiP one could simulate this using:

.. testcode:: [measurement]
    :hide:

    np.random.seed(42)

.. testcode:: [measurement]

   results = {1.0: 0, -1.0: 0}  # 1 and -1 are the possible outcomes
   for _ in range(1000):
      value, new_state = measure(up, spin_x)
      results[round(value)] += 1
   print(results)

**Output**:

.. testoutput:: [measurement]

   {1.0: 497, -1.0: 503}

which measures the x-component of the spin of the `up` state `1000` times and
stores the results in a dictionary. Afterwards we expect to have seen the
result `1.0` (i.e. left) roughly 500 times and the result `-1.0` (i.e. right)
roughly 500 times, but, of course, the number of each will vary slightly
each time we run it.

But what if we want to know the distribution of results precisely? In a
physical system, we would have to perform the measurement many many times,
but in QuTiP we can peak at the state itself and determine the probability
distribution of the outcomes exactly in a single line:
.. doctest:: [measurement]
    :hide:

   >>> np.random.seed(42)

.. doctest:: [measurement]

   >>> eigenvalues, eigenstates, probabilities = measurement_statistics(up, spin_x)

   >>> eigenvalues # doctest: +NORMALIZE_WHITESPACE
   array([-1., 1.])

   >>> eigenstates # doctest: +NORMALIZE_WHITESPACE
   array([Quantum object: dims = [[2], [1]], shape = (2, 1), type = ket
   Qobj data =
   [[ 0.70710678]
    [-0.70710678]],
          Quantum object: dims = [[2], [1]], shape = (2, 1), type = ket
   Qobj data =
   [[0.70710678]
    [0.70710678]]], dtype=object)

   >>> probabilities  # doctest: +NORMALIZE_WHITESPACE
   [0.5000000000000001, 0.4999999999999999]

The :func:`~qutip.measurement.measure` function returns three values:

* `eigenvalues` is an array of eigenvalues of the measurement operator, i.e.
  a list of the possible measurement results. In our example
  the value is `array([-1., -1.])`.

* `eigenstates` is an array of the eigenstates of the measurement operator, i.e.
  a list of the possible final states after the measurement is complete.
  Each element of the array is a :obj:`~qutip.Qobj`.

* `probabilities` is a list of the probabilities of each measurement result.
  In our example the value is `[0.5, 0.5]` since the `up` state has equal
  probability of being measured to be in the left (`-1.0`) or
  right (`1.0`) eigenstates.

All three lists are in the same order -- i.e. the first eigenvalue is
`eigenvalues[0]`, its corresponding eigenstate is `eigenstates[0]`, and
its probability is `probabilities[0]`, and so on.

The `measurement_statistics` function can provide statistics for measurements
of density matrices too. In this case `projectors` from the density matrix
onto the corresponding `eigenstates` are returned instead of the `eigenstates`.
You can read about these and other details at
:func:`~qutip.measurement.measurement_statistics`.
