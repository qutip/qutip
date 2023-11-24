.. _solver_api:



*******************************************
Solver Class Interface
*******************************************

Reusing Hamiltonian Data
------------------------


In order to solve a given simulation as fast as possible, the solvers in QuTiP
take the given input operators and break them down into simpler components before
passing them on to the ODE solvers. Although these operations are reasonably fast,
the time spent organizing data can become appreciable when repeatedly solving a
system over, for example, many different initial conditions. In cases such as
this, the Monte Carlo Solver may be reused after the initial configuration, thus
speeding up calculations.


Using the previous example, we will calculate the dynamics for two different
initial states, with the Hamiltonian data being reused on the second call

.. plot::
    :context: close-figs

    times = np.linspace(0.0, 10.0, 200)
    psi0 = tensor(fock(2, 0), fock(10, 5))
    a  = tensor(qeye(2), destroy(10))
    sm = tensor(destroy(2), qeye(10))

    H = 2*np.pi*a.dag()*a + 2*np.pi*sm.dag()*sm + 2*np.pi*0.25*(sm*a.dag() + sm.dag()*a)
    solver = MCSolver(H, c_ops=[np.sqrt(0.1) * a])
    data1 = solver.run(psi0, times, e_ops=[a.dag() * a, sm.dag() * sm], ntraj=100)
    psi1 = tensor(fock(2, 0), coherent(10, 2 - 1j))
    data2 = solver.run(psi1, times, e_ops=[a.dag() * a, sm.dag() * sm], ntraj=100)

    plt.figure()
    plt.plot(times, data1.expect[0], "b", times, data1.expect[1], "r", lw=2)
    plt.plot(times, data2.expect[0], 'b--', times, data2.expect[1], 'r--', lw=2)
    plt.title('Monte Carlo time evolution')
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Expectation values', fontsize=14)
    plt.legend(("cavity photon number", "atom excitation probability"))
    plt.show()

.. guide-dynamics-mc2:

The ``MCSolver`` also allows adding new trajectories after the first computation.
This is shown in the next example where the results of two separated runs with
identical conditions are merged into a single ``result`` object.

.. plot::
    :context: close-figs

    times = np.linspace(0.0, 10.0, 200)
    psi0 = tensor(fock(2, 0), fock(10, 5))
    a  = tensor(qeye(2), destroy(10))
    sm = tensor(destroy(2), qeye(10))

    H = 2*np.pi*a.dag()*a + 2*np.pi*sm.dag()*sm + 2*np.pi*0.25*(sm*a.dag() + sm.dag()*a)
    solver = MCSolver(H, c_ops=[np.sqrt(0.1) * a])
    data1 = solver.run(psi0, times, e_ops=[a.dag() * a, sm.dag() * sm], ntraj=1, seeds=1)
    data2 = solver.run(psi0, times, e_ops=[a.dag() * a, sm.dag() * sm], ntraj=1, seeds=3)
    data_merged = data1 + data2

    plt.figure()
    plt.plot(times, data1.expect[0], 'b-', times, data1.expect[1], 'g-', lw=2)
    plt.plot(times, data2.expect[0], 'b--', times, data2.expect[1], 'g--', lw=2)
    plt.plot(times, data_merged.expect[0], 'b:', times, data_merged.expect[1], 'g:', lw=2)
    plt.title('Monte Carlo time evolution')
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Expectation values', fontsize=14)
    plt.legend(("cavity photon number", "atom excitation probability"))
    plt.show()


This can be used to explore the convergence of the Monte Carlo solver.
For example, the following code block plots expectation values for 1, 10 and 100
trajectories:

.. plot::
    :context: close-figs

    solver = MCSolver(H, c_ops=[np.sqrt(0.1) * a])

    data1 = solver.run(psi0, times, e_ops=[a.dag() * a, sm.dag() * sm], ntraj=1)
    data10 = data1 + solver.run(psi0, times, e_ops=[a.dag() * a, sm.dag() * sm], ntraj=9)
    data100 = data10 + solver.run(psi0, times, e_ops=[a.dag() * a, sm.dag() * sm], ntraj=90)

    expt1 = data1.expect
    expt10 = data10.expect
    expt100 = data100.expect

    plt.figure()
    plt.plot(times, expt1[0], label="ntraj=1")
    plt.plot(times, expt10[0], label="ntraj=10")
    plt.plot(times, expt100[0], label="ntraj=100")
    plt.title('Monte Carlo time evolution')
    plt.xlabel('Time')
    plt.ylabel('Expectation values')
    plt.legend()
    plt.show()
