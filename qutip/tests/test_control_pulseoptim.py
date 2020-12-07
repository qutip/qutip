# -*- coding: utf-8 -*-
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2014 and later, Alexander J G Pitchford
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

# @author: Alexander Pitchford
# @email1: agp1@aber.ac.uk
# @email2: alex.pitchford@gmail.com
# @organization: Aberystwyth University
# @supervisor: Daniel Burgarth
# @date: Sep 2015

import pytest
import collections
import os
import pathlib
import tempfile
import numpy as np
import scipy.optimize

import qutip
from qutip.control import pulseoptim as cpo
from qutip.qip.algorithms import qft
from qutip.qip.operations.gates import hadamard_transform
import qutip.control.loadparams

_sx = qutip.sigmax()
_sy = qutip.sigmay()
_sz = qutip.sigmaz()
_sp = qutip.sigmap()
_sm = qutip.sigmam()
_si = qutip.identity(2)
_project_0 = qutip.basis(2, 0).proj()
_hadamard = hadamard_transform(1)

# We have a whole bunch of different physical systems we want to test the
# optimiser for, but the logic for testing them is largely the same.  To avoid
# having to explicitly parametrise over five linked parameters repeatedly, we
# group them into a record type, so that all the optimisation functions can
# then simply be parametrised over a single argument.
#
# We supply `kwargs` as a property of the system because the initial pulse type
# and dynamics solver to use vary, especially if the system is unitary.
_System = collections.namedtuple('_System',
                                 ['system', 'controls', 'initial', 'target',
                                  'kwargs'])

# Simple Hadamard gate.
_hadamard_kwargs = {'num_tslots': 10, 'evo_time': 10, 'gen_stats': True,
                    'init_pulse_type': 'LIN', 'fid_err_targ': 1e-10,
                    'dyn_type': 'UNIT'}
hadamard = _System(system=_sz,
                   controls=[_sx],
                   initial=_si,
                   target=_hadamard,
                   kwargs=_hadamard_kwargs)

# Quantum Fourier transform.
_qft_system = 0.5 * sum(qutip.tensor(op, op) for op in (_sx, _sy, _sz))
_qft_controls = [0.5*qutip.tensor(_sx, _si), 0.5*qutip.tensor(_sy, _si),
                 0.5*qutip.tensor(_si, _sx), 0.5*qutip.tensor(_si, _sy)]
_qft_kwargs = {'num_tslots': 10, 'evo_time': 10, 'gen_stats': True,
               'init_pulse_type': 'LIN', 'fid_err_targ': 1e-9,
               'dyn_type': 'UNIT'}
qft = _System(system=_qft_system,
              controls=_qft_controls,
              initial=qutip.identity([2, 2]),
              target=qft.qft(2),
              kwargs=_qft_kwargs)

# Coupling constants are completely arbitrary.
_ising_system = (0.9*qutip.tensor(_sx, _si) + 0.7*qutip.tensor(_si, _sx)
                 + 0.8*qutip.tensor(_sz, _si) + 0.9*qutip.tensor(_si, _sz))
_ising_kwargs = {'num_tslots': 10, 'evo_time': 18, 'init_pulse_type': 'LIN',
                 'fid_err_targ': 1e-10, 'dyn_type': 'UNIT'}
ising = _System(system=_ising_system,
                controls=[qutip.tensor(_sz, _sz)],
                initial=qutip.basis([2, 2], [0, 0]),
                target=qutip.basis([2, 2], [1, 1]),
                kwargs=_ising_kwargs)

# Louivillian amplitude-damping channel system.
_l_adc_system = 0.1 * (2*qutip.tensor(_sm, _sp.dag())
                       - qutip.tensor(_project_0, _si)
                       - qutip.tensor(_si, _project_0.dag()))
_l_adc_controls = [1j * (qutip.tensor(_si, _sz) - qutip.tensor(_sz, _si)),
                   1j * (qutip.tensor(_si, _sx) - qutip.tensor(_sx, _si))]
_l_adc_kwargs = {'num_tslots': 10, 'evo_time': 5, 'init_pulse_type': 'LIN',
                 'max_iter': 200, 'fid_err_targ': 1e-1, 'gen_stats': True}
l_adc = _System(system=_l_adc_system,
                controls=_l_adc_controls,
                initial=qutip.identity([2, 2]),
                target=hadamard_transform(2),
                kwargs=_l_adc_kwargs)

# Two coupled oscillators with symplectic dynamics.
_g1, _g2 = 1.0, 0.2
_A_rotate = qutip.qdiags([[1, 1, 0, 0]], [0])
_A_squeeze = 0.4 * qutip.qdiags([[1, -1, 0, 0]], [0])
_A_target = qutip.qdiags([[1, 1], [1, 1]], [2, -2])
_Omega = qutip.Qobj(qutip.control.symplectic.calc_omega(2))
_sympl_system = qutip.qdiags([[1, 1, 1, 1], [_g1, _g2], [_g1, _g2]],
                             [0, 2, -2])
_sympl_target = (-0.5 * _A_target * _Omega * np.pi).expm()
_sympl_kwargs = {'num_tslots': 20, 'evo_time': 10, 'fid_err_targ': 1e-3,
                 'max_iter': 200, 'dyn_type': 'SYMPL',
                 'init_pulse_type': 'ZERO', 'gen_stats': True}
symplectic = _System(system=_sympl_system,
                     controls=[_A_rotate, _A_squeeze],
                     initial=qutip.identity(4),
                     target=_sympl_target,
                     kwargs=_sympl_kwargs)


# Parametrise the systems and the propagation method separately so that we test
# all combinations of both.

# Test propagation with the default settings and with internal Qobj use for all
# test cases.
@pytest.fixture(params=[
    pytest.param(None, id="default propagation"),
    pytest.param({'oper_dtype': qutip.Qobj}, id="Qobj propagation"),
])
def propagation(request):
    return {'dyn_params': request.param}


# Any test requiring a system to test will parametrise over all of the ones we
# defined above.
@pytest.fixture(params=[
    pytest.param(hadamard, id="Hadamard gate"),
    pytest.param(qft, id="QFT"),
    pytest.param(ising, id="Ising state-to-state"),
    pytest.param(l_adc, id="Lindbladian amplitude damping channel"),
    pytest.param(symplectic, id="Symplectic coupled oscillators"),
])
def system(request):
    return request.param


def _optimize_pulse(system):
    """
    Unpack the `system` record type, optimise the result and assert that it
    succeeded.
    """
    result = cpo.optimize_pulse(system.system, system.controls,
                                system.initial, system.target,
                                **system.kwargs)
    error = " ".join(["Infidelity: {:7.4e}".format(result.fid_err),
                      "reason:", result.termination_reason])
    assert result.goal_achieved, error
    return result


def _merge_kwargs(system, kwargs):
    """
    Return a copy of `system` with any passed `kwargs` updated in the
    dictionary---this can be used to overwrite or to add new arguments.
    """
    out = system.kwargs.copy()
    out.update(kwargs)
    return system._replace(kwargs=out)


class TestOptimization:
    def test_basic_optimization(self, system, propagation):
        """Test the optimiser in the base case for each system."""
        system = _merge_kwargs(system, propagation)
        result = _optimize_pulse(system)
        assert result.fid_err < system.kwargs['fid_err_targ']

    def test_object_oriented_approach_and_gradient(self, system, propagation):
        """
        Test the object-oriented version of the optimiser, and ensure that the
        system truly appears to be at an extremum.
        """
        system = _merge_kwargs(system, propagation)
        base = _optimize_pulse(system)
        optimizer = cpo.create_pulse_optimizer(system.system, system.controls,
                                               system.initial, system.target,
                                               **system.kwargs)
        init_amps = np.array([optimizer.pulse_generator.gen_pulse()
                              for _ in system.controls]).T
        optimizer.dynamics.initialize_controls(init_amps)
        # Check the gradient numerically.
        func = optimizer.fid_err_func_wrapper
        grad = optimizer.fid_err_grad_wrapper
        loc = optimizer.dynamics.ctrl_amps.flatten()
        assert abs(scipy.optimize.check_grad(func, grad, loc)) < 1e-5,\
            "Gradient outside tolerance."
        result = optimizer.run_optimization()
        tol = system.kwargs['fid_err_targ']
        assert abs(result.fid_err-base.fid_err) < tol,\
            "Direct and indirect methods produce different results."

    @pytest.mark.parametrize("kwargs", [
        pytest.param({'gen_stats': False}, id="no stats"),
        pytest.param({'num_tslots': None, 'evo_time': None,
                      'tau': np.arange(1, 10, 1, dtype=np.float64)},
                     id="tau array")
    ])
    def test_modified_optimization(self, propagation, kwargs):
        """Test a basic system with a few different combinations of options."""
        system = _merge_kwargs(hadamard, kwargs)
        self.test_basic_optimization(system, propagation)

    def test_optimizer_bounds(self):
        """Test that bounds on the control fields are obeyed."""
        bound = 1.0
        kwargs = {'amp_lbound': -bound, 'amp_ubound': bound}
        system = _merge_kwargs(qft, kwargs)
        result = _optimize_pulse(system)
        assert np.all(result.final_amps >= -bound)
        assert np.all(result.final_amps <= bound)

    def test_unitarity_via_dump(self):
        """
        Test that unitarity is maintained at all times throughout the
        optimisation of the controls.
        """
        kwargs = {'num_tslots': 1000, 'evo_time': 4, 'fid_err_targ': 1e-9,
                  'dyn_params': {'dumping': 'FULL'}}
        system = _merge_kwargs(hadamard, kwargs)
        result = _optimize_pulse(system)
        dynamics = result.optimizer.dynamics
        assert dynamics.dump is not None, "Dynamics dump not created"
        # Use the dump to check unitarity of all propagators and evo_ops
        dynamics.unitarity_tol = 1e-13 # 1e-14 for eigh but 1e-13 for eig.
        for item, description in [('prop', 'propagators'),
                                  ('fwd_evo', 'forward evolution operators'),
                                  ('onto_evo', 'onto evolution operators')]:
            non_unitary = sum(not dynamics._is_unitary(x)
                              for dump in dynamics.dump.evo_dumps
                              for x in getattr(dump, item))
            assert non_unitary == 0, "Found non-unitary " + description + "."

    def test_crab(self, propagation):
        tol = 1e-5
        evo_time = 10
        result = cpo.opt_pulse_crab_unitary(
            hadamard.system, hadamard.controls,
            hadamard.initial, hadamard.target,
            num_tslots=12, evo_time=evo_time, fid_err_targ=tol,
            **propagation,
            alg_params={'crab_pulse_params': {'randomize_coeffs': False,
                                              'randomize_freqs': False}},
            init_coeff_scaling=0.5,
            guess_pulse_type='GAUSSIAN',
            guess_pulse_params={'variance': evo_time * 0.1},
            guess_pulse_scaling=1.0,
            guess_pulse_offset=1.0,
            amp_lbound=None,
            amp_ubound=None,
            ramping_pulse_type='GAUSSIAN_EDGE',
            ramping_pulse_params={'decay_time': evo_time * 0.01},
            gen_stats=True)
        error = " ".join(["Infidelity: {:7.4e}".format(result.fid_err),
                          "reason:", result.termination_reason])
        assert result.goal_achieved, error
        assert abs(result.fid_err) < tol
        assert abs(result.final_amps[0, 0]) < tol, "Lead-in amplitude nonzero."


# The full object-orientated interface to the optimiser is rather complex.  To
# attempt to simplify the test of the configuration loading, we break it down
# into steps here.

def _load_configuration(path):
    configuration = qutip.control.optimconfig.OptimConfig()
    configuration.param_fname = path.name
    configuration.param_fpath = str(path)
    configuration.pulse_type = "ZERO"
    qutip.control.loadparams.load_parameters(str(path), config=configuration)
    return configuration


def _load_dynamics(path, system, configuration, stats):
    dynamics = qutip.control.dynamics.DynamicsUnitary(configuration)
    dynamics.drift_dyn_gen = system.system
    dynamics.ctrl_dyn_gen = system.controls
    dynamics.initial = system.initial
    dynamics.target = system.target
    qutip.control.loadparams.load_parameters(str(path), dynamics=dynamics)
    dynamics.init_timeslots()
    dynamics.stats = stats
    return dynamics


def _load_pulse_generator(path, configuration, dynamics):
    pulse_generator = qutip.control.pulsegen.create_pulse_gen(
            pulse_type=configuration.pulse_type,
            dyn=dynamics)
    qutip.control.loadparams.load_parameters(str(path),
                                             pulsegen=pulse_generator)
    return pulse_generator


def _load_termination_conditions(path):
    conditions = qutip.control.termcond.TerminationConditions()
    qutip.control.loadparams.load_parameters(str(path), term_conds=conditions)
    return conditions


def _load_optimizer(path, configuration, dynamics, pulse_generator,
                    termination_conditions, stats):
    method = configuration.optim_method
    if method is None:
        raise qutip.control.errors.UsageError(
            "Optimization algorithm must be specified using the 'optim_method'"
            " parameter.")
    known = {'BFGS': 'OptimizerBFGS', 'FMIN_L_BFGS_B': 'OptimizerLBFGSB'}
    constructor = getattr(qutip.control.optimizer,
                          known.get(method, 'Optimizer'))
    optimizer = constructor(configuration, dynamics)
    optimizer.method = method
    qutip.control.loadparams.load_parameters(str(path), optim=optimizer)
    optimizer.config = configuration
    optimizer.dynamics = dynamics
    optimizer.pulse_generator = pulse_generator
    optimizer.termination_conditions = termination_conditions
    optimizer.stats = stats
    return optimizer


class TestFileIO:
    def test_load_parameters_from_file(self):
        system = hadamard
        path = pathlib.Path(__file__).parent / "Hadamard_params.ini"
        stats = qutip.control.stats.Stats()
        configuration = _load_configuration(path)
        dynamics = _load_dynamics(path, system, configuration, stats)
        pulse_generator = _load_pulse_generator(path, configuration, dynamics)
        termination_conditions = _load_termination_conditions(path)
        optimizer = _load_optimizer(path,
                                    configuration,
                                    dynamics,
                                    pulse_generator,
                                    termination_conditions,
                                    stats)
        init_amps = np.array([optimizer.pulse_generator.gen_pulse()
                              for _ in system.controls]).T
        optimizer.dynamics.initialize_controls(init_amps)
        result = optimizer.run_optimization()

        kwargs = {'num_tslots': 6, 'evo_time': 6, 'fid_err_targ': 1e-10,
                  'init_pulse_type': 'LIN', 'dyn_type': 'UNIT',
                  'amp_lbound': -1, 'amp_ubound': 1,
                  'gen_stats': True}
        target = _optimize_pulse(system._replace(kwargs=kwargs))
        np.testing.assert_allclose(result.final_amps, target.final_amps,
                                   atol=1e-5)

    @pytest.mark.usefixtures("in_temporary_directory")
    def test_dumping_to_files(self):
        N_OPTIMDUMP_FILES = 10
        N_DYNDUMP_FILES = 49
        dumping = {'dumping': 'FULL', 'dump_to_file': True}
        kwargs = {'num_tslots': 1_000, 'evo_time': 4, 'fid_err_targ': 1e-9,
                  'optim_params': {'dump_dir': 'optim', **dumping},
                  'dyn_params': {'dump_dir': 'dyn', **dumping}}
        system = _merge_kwargs(hadamard, kwargs)
        result = _optimize_pulse(system)

        # Check dumps were generated and have the right number of files.
        assert result.optimizer.dump is not None
        assert result.optimizer.dynamics.dump is not None
        assert (len(os.listdir(result.optimizer.dump.dump_dir))
                == N_OPTIMDUMP_FILES)
        assert (len(os.listdir(result.optimizer.dynamics.dump.dump_dir))
                == N_DYNDUMP_FILES)

        # Dump all to specific file stream.
        for dump, type_ in [(result.optimizer.dump, 'optimizer'),
                            (result.optimizer.dynamics.dump, 'dynamics')]:
            with tempfile.NamedTemporaryFile() as file:
                dump.writeout(file)
                assert os.stat(file.name).st_size > 0,\
                    " ".join(["Empty", type_, "file."])


def _count_waves(system):
    optimizer = cpo.create_pulse_optimizer(system.system, system.controls,
                                           system.initial, system.target,
                                           **system.kwargs)
    pulse = optimizer.pulse_generator.gen_pulse()
    zero_crossings = pulse[0:-2]*pulse[1:-1] < 0
    return (sum(zero_crossings) + 1) // 2


@pytest.mark.parametrize('pulse_type',
                         [pytest.param(x, id=x.lower())
                          for x in ['SINE', 'SQUARE', 'TRIANGLE', 'SAW']])
class TestPeriodicControlFunction:
    num_tslots = 1_000
    evo_time = 10

    @pytest.mark.parametrize('n_waves', [1, 5, 10, 100])
    def test_number_of_waves(self, pulse_type, n_waves):
        kwargs = {'num_tslots': self.num_tslots, 'evo_time': self.evo_time,
                  'init_pulse_type': pulse_type,
                  'init_pulse_params': {'num_waves': n_waves},
                  'gen_stats': False}
        system = _merge_kwargs(hadamard, kwargs)
        assert _count_waves(system) == n_waves

    @pytest.mark.parametrize('frequency', [0.1, 1, 10, 20])
    def test_frequency(self, pulse_type, frequency):
        kwargs = {'num_tslots': self.num_tslots, 'evo_time': self.evo_time,
                  'init_pulse_type': pulse_type,
                  'init_pulse_params': {'freq': frequency},
                  'fid_err_targ': 1e-5,
                  'gen_stats': False}
        system = _merge_kwargs(hadamard, kwargs)
        assert _count_waves(system) == self.evo_time*frequency


class TestTimeDependence:
    """
    Test that systems where the system Hamiltonian is time-dependent behave as
    expected under the optimiser.
    """
    def test_drift(self):
        """
        Test that introducing time dependence to the system does change the
        result of the optimisation.
        """
        num_tslots = 20
        system = _merge_kwargs(hadamard, {'num_tslots': num_tslots,
                                          'evo_time': 10})
        result_fixed = _optimize_pulse(system)
        system_flat = system._replace(system=[system.system]*num_tslots)
        result_flat = _optimize_pulse(system_flat)
        step = [0.0]*(num_tslots//2) + [1.0]*(num_tslots//2)
        system_step = system._replace(system=[x*system.system for x in step])
        result_step = _optimize_pulse(system_step)
        np.testing.assert_allclose(result_fixed.final_amps,
                                   result_flat.final_amps,
                                   rtol=1e-9)
        assert np.any((result_flat.final_amps-result_step.final_amps) > 1e-3),\
            "Flat and step drights result in the same control pulses."

    def test_controls_all_time_slots_equal_to_no_time_dependence(self):
        """
        Test that simply duplicating the system in each time slot (i.e. no
        actual time dependence has no effect on the final result.
        """
        num_tslots = 20
        system = _merge_kwargs(hadamard, {'num_tslots': num_tslots,
                                          'evo_time': 10,
                                          'fid_err_targ': 1e-10})
        result_single = _optimize_pulse(system)
        system_vary = system._replace(controls=[[_sx]]*num_tslots)
        result_vary = _optimize_pulse(system_vary)
        np.testing.assert_allclose(result_single.final_amps,
                                   result_vary.final_amps,
                                   atol=1e-9)

    def test_controls_identity_operators_ignored(self):
        """
        Test that moments in time where the control parameters are simply the
        identity are just ignored by the optimiser (since they'll never be able
        to do anything.
        """
        num_tslots = 20
        controls = [[_sx] if k % 3 else [_si] for k in range(num_tslots)]
        system = _merge_kwargs(hadamard, {'num_tslots': num_tslots,
                                          'evo_time': 10})
        system = system._replace(controls=controls)
        result = _optimize_pulse(system)
        for k in range(0, num_tslots, 3):
            np.testing.assert_allclose(result.initial_amps[k],
                                       result.final_amps[k],
                                       rtol=1e-9)
