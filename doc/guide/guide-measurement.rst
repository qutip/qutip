.. _measurement:

******************************
Measurement of Quantum Objects
******************************

.. note::
   New in QuTiP 4.6

.. _measurement-intro:

Introduction
------------

Measurement is a fundamental part of the standard formulation of quantum
mechanics and is the process by which classical readings are obtained from
a quantum object. Although the interpretation of the procedure is at times
contentious, the procedure itself is mathematically straightforward and is
described in many good introductory texts.

Here we will show you how to perform simple measurement operations on QuTiP
objects. The same functions :func:`~qutip.measurement.measure` and
:func:`~qutip.measurement.measurement_statistics` can be used
to handle both observable-style measurements and projective style measurements.

.. _measurement-basic:

Performing a basic measurement (Observable)
-------------------------------------------

First we need to select some states to measure. For now, let us create an *up*
state and a *down* state:

.. testcode::

   up = basis(2, 0)

   down = basis(2, 1)

which represent spin-1/2 particles with their spin pointing either up or down
along the z-axis.

We choose what to measure (in this case) by selecting a **measurement operator**.
For example,
we could select :func:`.sigmaz` which measures the z-component of the
spin of a spin-1/2 particle, or :func:`.sigmax` which measures the
x-component:

.. testcode::

   spin_z = sigmaz()

   spin_x = sigmax()

How do we know what these operators measure? The answer lies in the measurement
procedure itself:

* A quantum measurement transforms the state being measured by projecting it into
  one of the eigenvectors of the measurement operator.

* Which eigenvector to project onto is chosen probabilistically according to the
  square of the amplitude of the state in the direction of the eigenvector.

* The value returned by the measurement is the eigenvalue corresponding to the
  chosen eigenvector.

.. note::

   How to interpret this "random choosing" is the famous
   "quantum measurement problem".

The eigenvectors of `spin_z` are the states with their spin pointing either up
or down, so it measures the component of the spin along the z-axis.

The eigenvectors of `spin_x` are the states with their spin pointing either
left or right, so it measures the component of the spin along the x-axis.

When we measure our `up` and `down` states using the operator `spin_z`, we
always obtain:

.. testcode::

   from qutip.measurement import measure, measurement_statistics

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

.. testcode::

   measure(up, spin_x)

The `up` state is not an eigenvector of `spin_x`. `spin_x` has two eigenvectors
which we will call `left` and `right`. The `up` state has equal components in
the direction of these two vectors, so measurement will select each of them
50% of the time.

These `left` and `right` states are:

.. testcode::

   left = (up - down).unit()

   right = (up + down).unit()

When `left` is chosen, the result of the measurement will be `(-1.0, -left)`.

When `right` is chosen, the result of measurement with be `(1.0, right)`.

.. note::

  When :func:`~qutip.measurement.measure` is invoked with the second argument
  being an observable, it acts as an alias to
  :func:`~qutip.measurement.measure_observable`.

Performing a basic measurement (Projective)
-------------------------------------------

We can also choose what to measure by specifying a *list of projection operators*. For
example, we could select the projection operators :math:`\ket{0} \bra{0}` and
:math:`\ket{1} \bra{1}` which measure the state in the :math:`\ket{0}, \ket{1}`
basis. Note that these projection operators are simply the projectors determined by
the eigenstates of the :func:`~qutip.operators.sigmaz` operator.

.. testcode::

   Z0, Z1 = ket2dm(basis(2, 0)), ket2dm(basis(2, 1))

The probabilities and respective output state
are calculated for each projection operator.

.. testcode::

   measure(up, [Z0, Z1]) == (0, up)

   measure(down, [Z0, Z1]) == (1, down)

In this case, the projection operators are conveniently eigenstates corresponding
to subspaces of dimension :math:`1`. However, this might not be
the case, in which case it is not possible to have unique eigenvalues for each
eigenstate. Suppose we want to measure only the first
qubit in a two-qubit system. Consider the two qubit state :math:`\ket{0+}`

.. testcode::

   state_0 = basis(2, 0)

   state_plus = (basis(2, 0) + basis(2, 1)).unit()

   state_0plus = tensor(state_0, state_plus)

Now, suppose we want to measure only the first qubit in the computational basis.
We can do that by measuring with the projection operators
:math:`\ket{0}\bra{0} \otimes I` and  :math:`\ket{1}\bra{1} \otimes I`.

.. testcode::

   PZ1 = [tensor(Z0, identity(2)), tensor(Z1, identity(2))]

   PZ2 = [tensor(identity(2), Z0), tensor(identity(2), Z1)]

Now, as in the previous example, we can measure by supplying a list of projection operators
and the state.

.. testcode::

    measure(state_0plus, PZ1) == (0, state_0plus)

The output of the measurement is the index of the measurement outcome as well
as the output state on the full Hilbert space of the input state. It is crucial to
note that we do not discard the measured qubit after measurement (as opposed to
when measuring on quantum hardware).

.. note::

  When :func:`~qutip.measurement.measure` is invoked with the second argument
  being a list of projectors, it acts as an alias to
  :func:`~qutip.measurement.measure_povm`.

The :func:`~qutip.measurement.measure` function can perform measurements on
density matrices too. You can read about these and other details at
:func:`~qutip.measurement.measure_povm` and :func:`~qutip.measurement.measure_observable`.

Now you know how to measure quantum states in QuTiP!

.. _measurement-statistics:

Obtaining measurement statistics(Observable)
--------------------------------------------

You've just learned how to perform measurements in QuTiP, but you've also
learned that measurements are probabilistic. What if instead of just making
a single measurement, we want to determine the probability distribution of
a large number of measurements?

One way would be to repeat the measurement many times -- and this is what
happens in many quantum experiments. In QuTiP one could simulate this using:

.. testcode::
    :hide:

    np.random.seed(42)

.. testcode::

   results = {1.0: 0, -1.0: 0}  # 1 and -1 are the possible outcomes
   for _ in range(1000):
      value, new_state = measure(up, spin_x)
      results[round(value)] += 1
   print(results)

**Output**:

.. testoutput::

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

.. doctest::
    :hide:

    >>> np.random.seed(42)

.. doctest::

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

The :func:`~qutip.measurement.measurement_statistics` function then returns three values
when called with a single observable:

- `eigenvalues` is an array of eigenvalues of the measurement operator, i.e.
  a list of the possible measurement results. In our example
  the value is `array([-1., -1.])`.

- `eigenstates` is an array of the eigenstates of the measurement operator, i.e.
  a list of the possible final states after the measurement is complete.
  Each element of the array is a :obj:`.Qobj`.

- `probabilities` is a list of the probabilities of each measurement result.
  In our example the value is `[0.5, 0.5]` since the `up` state has equal
  probability of being measured to be in the left (`-1.0`) or
  right (`1.0`) eigenstates.

All three lists are in the same order -- i.e. the first eigenvalue is
`eigenvalues[0]`, its corresponding eigenstate is `eigenstates[0]`, and
its probability is `probabilities[0]`, and so on.

.. note::

   When :func:`~qutip.measurement.measurement_statistics`
   is invoked with the second argument
   being an observable, it acts as an alias to
   :func:`~qutip.measurement.measurement_statistics_observable`.


Obtaining measurement statistics(Projective)
--------------------------------------------

Similarly, when we want to obtain measurement statistics for projection operators,
we can use the `measurement_statistics` function with the second argument being a list of projectors.
Consider again, the state :math:`\ket{0+}`.
Suppose, now we want to obtain the measurement outcomes for the second qubit. We
must use the projectors specified earlier by `PZ2` which allow us to measure only
on the second qubit. Since the second qubit has the state :math:`\ket{+}`, we get
the following result.

.. testcode::

   collapsed_states, probabilities = measurement_statistics(state_0plus, PZ2)

   print(collapsed_states)

**Output**:

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   [Quantum object: dims = [[2, 2], [1, 1]], shape = (4, 1), type = ket
    Qobj data =
    [[1.]
     [0.]
     [0.]
     [0.]], Quantum object: dims = [[2, 2], [1, 1]], shape = (4, 1), type = ket
    Qobj data =
    [[0.]
     [1.]
     [0.]
     [0.]]]

.. testcode::

   print(probabilities)

**Output**:

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   [0.4999999999999999, 0.4999999999999999]

The function :func:`~qutip.measurement.measurement_statistics` then returns two values:

* `collapsed_states` is an array of the possible final states after the
  measurement is complete. Each element of the array is a :obj:`.Qobj`.

* `probabilities` is a list of the probabilities of each measurement outcome.

Note that the collapsed_states are exactly :math:`\ket{00}` and :math:`\ket{01}`
with equal probability, as expected. The two lists are in the same order.

.. note::

   When :func:`~qutip.measurement.measurement_statistics`
   is invoked with the second argument
   being a list of projectors, it acts as an alias to
   :func:`~qutip.measurement.measurement_statistics_povm`.

The :func:`~qutip.measurement.measurement_statistics` function can provide statistics for measurements
of density matrices too.
You can read about these and other details at
:func:`~qutip.measurement.measurement_statistics_observable`
and :func:`~qutip.measurement.measurement_statistics_povm`.

Furthermore, the :func:`~qutip.measurement.measure_povm`
and :func:`~qutip.measurement.measurement_statistics_povm` functions can
handle POVM measurements which are more general than projective measurements.
