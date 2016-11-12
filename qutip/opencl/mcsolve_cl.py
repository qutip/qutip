# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2016 and later, Christian Wasserthal.
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

__all__ = ['mcsolve_cl', 'list_opencl_platforms']

import time
import numpy as np
from qutip.qobj import Qobj
from qutip.solver import Options, Result
from qutip.settings import debug
import pyopencl as cl
import logging
import six


if debug:
    import inspect


def mcsolve_cl(H, psi0, tlist, c_ops, e_ops, ntraj=None, options=None,
               args={}, platform=None, double_prec=True):
    """Monte Carlo evolution of a state vector :math:`|\psi \\rangle` for a
    given Hamiltonian and sets of collapse operators, and possibly, operators
    for calculating expectation values. Options for the underlying ODE solver
    are given by the Options class.

    In many cases this function can be used as a drop in replacement for mcsolve.
    It supports time-dependent Hamiltonians using strings. The strings become
    part of the OpenCL program and are limited to the
    `math functions available in OpenCL <https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/mathFunctions.html>`.

    Since OpenCL does not support complex numbers natively, complex 
    coefficients can supplied as a tuple. For instance:
    ``mcsolve([H1, [H12, '2+3j*sin(omega*t)']], ...)``
    becomes
    ``mcsolve_cl([H1, [H12, ('2.0', '3.0*sin(omega*t)')]], ...)``.
    And likewise for collapse operators. If the imaginary part is zero, a
    string can be used instead of a tuple.

    Parameters
    ----------
    H : :class:`Qobj`
        System Hamiltonian.

    psi0 : :class:`Qobj`
        Initial state vector

    tlist : array_like
        Times at which results are recorded.

    ntraj : int
        Number of trajectories to run.

    c_ops : array_like
        single collapse operator or ``list`` or ``array`` of collapse
        operators.

    e_ops : array_like
        single operator or ``list`` or ``array`` of operators for calculating
        expectation values.

    options : :class:`Options`
        Instance of solver options.

    args : dict
        Arguments for time-dependent Hamiltonian and collapse operator terms.

    double_prec : bool
        If enabled, use double precision.

    platform : int
        Selects the OpenCL platform. See also :func:`list_opencl_platforms`.

    Returns
    -------
    results : :class:`Result`
        Object storing all results from the simulation.

    .. note::

        Random seeds from other solvers should not be used in mcsolve_cl.

        For an atol below 1e-5, double precision may be required. With
        double_prec=False and options=None an atol of 1e-5 will be used,
        leaving all other options at their default values.

        It is possible to reuse the random number seeds from a previous run
        of the mcsolve_cl by passing the output Result object seeds via the
        Options class, i.e. Options(seeds=prev_result.seeds).

        mcsolve_cl always uses the dopri method for integration.
        The following options have no effect for mcsolve_cl: rtol, method,
        order, rhs_with_state, rhs_reuse, rhs_filename, mc_corr_eps,
        num_cpus, norm_steps.
    """

    if debug:
        print(inspect.stack()[0][3])

    logger = logging.getLogger(__name__)

    if options is None:
        options = Options()
        if not double_prec:
            options.atol = 1e-5

    if ntraj is None:
        ntraj = options.ntraj

    ntraj_int = ntraj
    if isinstance(ntraj, (list, np.ndarray)):
        ntraj_int = max(ntraj)

    if not psi0.isket:
        raise Exception("Initial state must be a state vector.")

    # check e_ops and c_ops
    if isinstance(c_ops, Qobj):
        c_ops = [c_ops]

    if isinstance(e_ops, Qobj):
        e_ops = [e_ops]

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    # check H
    if isinstance(H, Qobj):
        H = [H]

    assert isinstance(c_ops, list)
    assert isinstance(e_ops, list)
    assert isinstance(H, list)

    # for compatibility with mcsolve
    if len(e_ops) == 0:
        options.store_states = True

    # for compatibility with mcsolve
    if len(c_ops) == 0:
        ntraj = ntraj_int = 1

    logger.debug("ntraj_int = %d" % ntraj_int)

    # generate random seeds
    if options.seeds is None:
        seeds = np.frombuffer(np.random.bytes(ntraj_int * 8), dtype=np.int64)
    elif len(options.seeds) > ntraj_int:
        seeds = options.seeds[0:ntraj_int].astype(np.int64, copy=False)
    elif len(options.seeds) < ntraj_int:
        newseeds = np.frombuffer(
            np.random.bytes((ntraj_int - len(options.seeds)) * 8), dtype=np.int64)
        seeds = np.hstack((options.seeds.astype(np.int64, copy=False), newseeds))
    else:
        seeds = options.seeds

    assert seeds.dtype == np.int64
    assert len(seeds) == ntraj_int

    num_times = len(tlist)
    dim = len(psi0.full())

    # set dtype for real and complex numbers
    if double_prec:
        stype = np.float64
        ctype = np.complex128
    else:
        stype = np.float32
        ctype = np.complex64

    # set dtype for vector indices
    if dim - 1 < 256:
        idxtype = np.uint8
    elif dim - 1 < 65536:
        idxtype = np.uint16
    else:
        idxtype = np.uint32

    # create copies so we don't modify input objects
    H = list(H)
    c_ops = list(c_ops)

    # combine constant H terms into one matrix
    H_const = 0
    time_dependent = False
    for i in range(len(H)):
        _check_operator_format(H[i])
        # the Hamiltonians are premultiplied with -1j
        if isinstance(H[i], Qobj):
            H_const -= 1j * H[i]
        else:
            H[i] = list(H[i])
            H[i][0] = -1j * H[i][0]
            time_dependent = True
    for i in range(len(c_ops)):
        _check_operator_format(c_ops[i])
        if isinstance(c_ops[i], Qobj):
            # the constant collapse operators that become part of the
            # effective Hamiltonian are premultiplied with -0.5
            H_const -= 0.5 * c_ops[i].dag() * c_ops[i]
        else:
            c_ops[i] = list(c_ops[i])
            time_dependent = True

    # Note: The summands of H_const have been pre-multiplied in such a way
    # that without time-dependent operators
    # psi(t+dt) = psi(t) + H_const*psi(t)*dt.
    # Time-dependent Hamiltonians are premultiplied with -1j. The collapse
    # operators remain unchanged.

    # remove constant terms in H
    H = [Hi for Hi in H if not isinstance(Hi, Qobj)]

    # prepend H_const to H
    H = [H_const] + H

    logger.debug("time_depenent = %s" % time_dependent)

    # create host arrays
    np_seeds = seeds
    np_tlist = tlist.astype(stype, copy=False)
    np_psi0 = psi0.full().ravel().astype(ctype, copy=False)
    np_data = []
    np_ind = []
    np_ptr = []
    if options.store_states:
        np_states_out = np.full(dim * ntraj_int * num_times, np.nan, dtype=ctype)
    elif options.store_final_state:
        np_states_out = np.full(dim * ntraj_int, np.nan, dtype=ctype)
    else:
        np_states_out = None
    np_expect_out = np.full(ntraj_int * len(e_ops) * num_times, np.nan, dtype=ctype) \
        if len(e_ops) > 0 else None
    np_status_out = np.zeros(ntraj_int, dtype=np.int8)
    np_shared_mem = ntraj_int * 9 * dim * ctype().nbytes

    # The sparse matrices that describe the operators are stored as three
    # arrays: ptr, data and ind.
    # The ptr array has always the length len(psi0)+1 and it's values are
    # indices to data and ind (they have the same length).

    # Here we concatenate the ptr, data and ind arrays of all matrices needed
    # for the computation and correct their indices to match their new position
    # in these large arrays.

    offset = 0
    for Hi in H:
        if isinstance(Hi, list) or isinstance(Hi, tuple):
            Hi = Hi[0]
        assert isinstance(Hi, Qobj)
        if options.tidy:
            Hi = Hi.tidyup(options.atol)
        assert len(Hi.data.indptr) == dim + 1
        assert len(Hi.data.data) == len(Hi.data.indices)
        np_ptr.append(Hi.data.indptr + offset)
        np_ind.append(Hi.data.indices.astype(idxtype))
        np_data.append(Hi.data.data.astype(ctype))
        offset += len(Hi.data.data)
    for c_op in c_ops:
        if isinstance(c_op, list) or isinstance(c_op, tuple):
            c_op = c_op[0]
        assert isinstance(c_op, Qobj)
        n_op = c_op.dag() * c_op
        if options.tidy:
            n_op = n_op.tidyup(options.atol)
        assert len(n_op.data.indptr) == dim + 1
        assert len(n_op.data.data) == len(n_op.data.indices)
        np_ptr.append(n_op.data.indptr + offset)
        np_ind.append(n_op.data.indices.astype(idxtype))
        np_data.append(n_op.data.data.astype(ctype))
        offset += len(n_op.data.data)
    for c_op in c_ops:
        if isinstance(c_op, list) or isinstance(c_op, tuple):
            c_op = c_op[0]
        assert isinstance(c_op, Qobj)
        if options.tidy:
            c_op = c_op.tidyup(options.atol)
        assert len(c_op.data.indptr) == dim + 1
        assert len(c_op.data.data) == len(c_op.data.indices)
        np_ptr.append(c_op.data.indptr + offset)
        np_ind.append(c_op.data.indices.astype(idxtype))
        np_data.append(c_op.data.data.astype(ctype))
        offset += len(c_op.data.data)
    for e_op in e_ops:
        assert isinstance(e_op, Qobj)
        if options.tidy:
            e_op = e_op.tidyup(options.atol)
        assert len(e_op.data.indptr) == dim + 1
        assert len(e_op.data.data) == len(e_op.data.indices)
        np_ptr.append(e_op.data.indptr + offset)
        np_ind.append(e_op.data.indices.astype(idxtype))
        np_data.append(e_op.data.data.astype(ctype))
        offset += len(e_op.data.data)

    np_data = np.concatenate(np_data)
    np_ind = np.concatenate(np_ind)
    np_ptr = np.concatenate(np_ptr)

    assert len(np_ptr) == (dim + 1) * (len(H) + 2 * len(c_ops) + len(e_ops))

    # set dtype for the ptr array
    ptr_max = np.max(np_ptr)
    if ptr_max < 256:
        ptrtype = np.uint8
    elif ptr_max < 65536:
        ptrtype = np.uint16
    else:
        ptrtype = np.uint32
    np_ptr = np_ptr.astype(ptrtype)

    logger.debug("max idx = %d --> using idxtype = " % np.max(np_ind) + idxtype.__name__)
    logger.debug("max ptr = %d --> using ptrtype = " % np.max(np_ptr) + ptrtype.__name__)

    # compute estimate of opencl buffer sizes (for debugging)
    size_in = 0
    for x in [np_seeds, np_tlist, np_psi0, np_data, np_ind, np_ptr]:
        size_in += x.nbytes
    size_out = np_status_out.nbytes
    if options.store_states or options.store_final_state:
        size_out += np_states_out.nbytes
    if len(e_ops) > 0:
        size_out += np_expect_out.nbytes
    logger.info("size of input buffers: %d bytes" % size_in)
    logger.info("size of output buffers: %d bytes" % size_out)
    logger.info("size of shared mem: %d bytes" % np_shared_mem)

    # create opencl context
    if platform is None:
        context = cl.create_some_context()
    else:
        context = cl.Context(cl.get_platforms()[platform].get_devices())

    logger.info("using OpenCL platform: " + context.devices[0].platform.name)

    if double_prec:
        for d in context.devices:
            if d.double_fp_config == 0:
                raise Exception("The selected OpenCL platform does not support double precision. Please set double_prec=False or choose a different platform.")

    # create device buffers
    mf = cl.mem_flags
    buf_seeds = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_seeds)
    buf_tlist = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_tlist)
    buf_psi0 = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_psi0)
    buf_data = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_data)
    buf_ind = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_ind)
    buf_ptr = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_ptr)
    buf_states_out = cl.Buffer(context, mf.WRITE_ONLY, np_states_out.nbytes) \
        if options.store_states or options.store_final_state else None
    buf_expect_out = cl.Buffer(context, mf.WRITE_ONLY, np_expect_out.nbytes) \
        if len(e_ops) > 0 else None
    buf_status_out = cl.Buffer(context, mf.WRITE_ONLY, np_status_out.nbytes)
    buf_shared_mem = cl.Buffer(context, mf.READ_WRITE, np_shared_mem)

    # configure program
    programcfg = {
        'ntraj': ntraj_int,
        'H_len': len(H),
        'c_ops_len': len(c_ops),
        'e_ops_len': len(e_ops),
        'idxtype': _nptype2cl(idxtype),
        'ptrtype': _nptype2cl(ptrtype),
        'stype': _nptype2cl(stype),
        'ctype': _nptype2cl(ctype),
        'len': dim,
        'num_times': num_times,
        'hstart': options.first_step if options.first_step else 0.1,
        'hmin': options.min_step if options.min_step else 1e-9,
        'hmax': options.max_step if options.max_step else 1.0,
        'nsteps': options.nsteps,
        'atol2': options.atol**2,
        'norm_rtol': options.norm_tol,
        'WG_X': ntraj_int,
        'WG_Y': 1,
        'WG_Z': 1,
    }

    programcfg.update(args)

    cl_src = '//CL//\n\n'

    if double_prec:
        cl_src += '#pragma OPENCL EXTENSION cl_khr_fp64: enable\n\n'

    for name, value in programcfg.items():
        if isinstance(value, str):
            cl_src += "#define " + name + " " + value + "\n"
        else:
            cl_src += "#define " + name + " (" + str(value) + ")\n"

    cl_src += r"""
#define a21 (1.0/5.0)
#define a31 (3.0/40.0)
#define a32 (9.0/40.0)
#define a41 (44.0/45.0)
#define a42 (-56.0/15.0)
#define a43 (32.0/9.0)
#define a51 (19372.0/6561.0)
#define a52 (-25360.0/2187.0)
#define a53 (64448.0/6561.0)
#define a54 (-212.0/729.0)
#define a61 (9017.0/3168.0)
#define a62 (-355.0/33.0)
#define a63 (46732.0/5247.0)
#define a64 (49.0/176.0)
#define a65 (-5103.0/18656.0)
#define a71 (35.0/384.0)
#define a72 (0.0)
#define a73 (500.0/1113.0)
#define a74 (125.0/192.0)
#define a75 (-2187.0/6784.0)
#define a76 (11.0/84.0)
#define c1  (0.0)
#define c2  (1.0/5.0)
#define c3  (3.0/10.0)
#define c4  (4.0/5.0)
#define c5  (8.0/9.0)
#define c6  (1.0)
#define c7  (1.0)
#define b1  (35.0/384.0)
#define b2  (0.0)
#define b3  (500.0/1113.0)
#define b4  (125.0/192.0)
#define b5  (-2187.0/6784.0)
#define b6  (11.0/84.0)
#define b7  (0.0)
#define b1p (5179.0/57600.0)
#define b2p (0.0)
#define b3p (7571.0/16695.0)
#define b4p (393.0/640.0)
#define b5p (-92097.0/339200.0)
#define b6p (187.0/2100.0)
#define b7p (1.0/40.0)

ctype cmul(const ctype a, const ctype b){
  return (ctype)( a.s0*b.s0 - a.s1*b.s1 , a.s0*b.s1 + a.s1*b.s0 );
}

// conj(a)*b
ctype conjmul(const ctype a, const ctype b){
  return (ctype)( a.s0*b.s0 + a.s1*b.s1 , a.s0*b.s1 - a.s1*b.s0 );
}

stype cabs2(const ctype x)
{
  return x.s0*x.s0 + x.s1*x.s1;
}

stype sqr(const stype a){
  return a*a;
}

stype norm_sqr(global const ctype *state){
  stype len_sqr = 0.0f;
  for(idxtype j=0;j<len;j++){
    len_sqr += sqr(state[j].s0) + sqr(state[j].s1);
  }
  return len_sqr;
}

void spmv_csr(
  constant const ctype *data,
  constant const idxtype *idx,
  constant const ptrtype *ptr,
  global const ctype *vec,
  global ctype *out)
{
  for(idxtype row=0;row<len;row++){
    ctype dot = 0.0f;
    const ptrtype row_start = ptr[row];
    const ptrtype row_end = ptr[row+1];
    for(ptrtype jj=row_start;jj<row_end;jj++){
      dot += cmul(data[jj],vec[idx[jj]]);
    }
    out[row] = dot;
  }
}

void spmv_csr_add(
  constant const ctype *data,
  constant const idxtype *idx,
  constant const ptrtype *ptr,
  global const ctype *vec,
  global ctype *out,
  const ctype factor)
{
  for(idxtype row=0;row<len;row++){
    ctype dot = 0.0f;
    const ptrtype row_start = ptr[row];
    const ptrtype row_end = ptr[row+1];
    for(ptrtype jj=row_start;jj<row_end;jj++){
      dot += cmul(data[jj],vec[idx[jj]]);
    }
    out[row] += cmul(factor,dot);
  }
}

void integrand(
  constant const ctype *data,
  constant const idxtype *idx,
  constant const ptrtype *ptr,
  const stype t,
  global const ctype *vec,
  global ctype *out)
{
  spmv_csr(data,idx,ptr,vec,out);"""

    # add time-dependent H terms
    for i, Hi in enumerate(H):
        if isinstance(Hi, (list, tuple)):
            if isinstance(Hi[1], six.string_types):
                cl_src += "\n  spmv_csr_add(data,idx,ptr+(len+1)*%d,vec,out,(ctype)(%s,0.0));" % (i, Hi[1])
            else:
                cl_src += "\n  spmv_csr_add(data,idx,ptr+(len+1)*%d,vec,out,(ctype)(%s,%s));" % (i, Hi[1][0], Hi[1][1])

    # add time-dependent c_ops
    for i, c_op in enumerate(c_ops):
        if isinstance(c_op, (list, tuple)):
            if isinstance(c_op[1], six.string_types):
                cl_src += "\n  spmv_csr_add(data,idx,ptr+(len+1)*(H_len+%d),vec,out,-0.5*(sqr(%s)));" % (i, c_op[1])
            else:
                cl_src += "\n  spmv_csr_add(data,idx,ptr+(len+1)*(H_len+%d),vec,out,-0.5*(sqr(%s)+sqr(%s)));" % (i, c_op[1][0], c_op[1][1])

    cl_src += r"""
}

ctype cy_expect_psi_csr(
  constant const ctype *data,
  constant const idxtype *idx,
  constant const ptrtype *ptr,
  global const ctype *state)
{
  ctype dot = 0.0;
  for(idxtype row=0;row<len;row++){
    ctype tmp = 0.0;
    const ptrtype row_start = ptr[row];
    const ptrtype row_end = ptr[row+1];
    for(ptrtype jj=row_start;jj<row_end;jj++){
      tmp += cmul(data[jj],state[idx[jj]]);
    }
    dot += conjmul(state[row],tmp);
  }
  return dot;
}

// http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
float random(uint2 *state){
  enum {A=4294883355U};
  uint x=(*state).x, c=(*state).y;
  uint res=x^c;
  uint hi=mul_hi(x,A);
  x=x*A+c;
  c=hi+(x<c);
  *state=(uint2)(x,c);
  return res/4294967296.0f;
}

kernel void mcsolve(
    constant const uint2* restrict seeds,
    constant const stype* restrict tlist,
    constant const ctype* restrict psi0,
    constant const ctype* restrict data,
    constant const idxtype* restrict ind,
    constant const ptrtype* restrict ptr,"""

    if options.store_states or options.store_final_state:
        cl_src += "\n    global ctype* restrict states_out,"

    if len(e_ops) > 0:
        cl_src += "\n    global ctype* restrict expect_out,"

    cl_src += r"""
    global char* restrict status_out,
    global ctype* restrict shared_mem
){
    const uint traj = get_global_id(0);
    uint2 seed = seeds[traj];

    /*
     * These are temporary vectors used in the runge-kutta integration.
     * They should be allocated in private memory, but the author ran into
     * too many problems with various OpenCL implementations.
     * Since they do not fit in the private memory anyway, they are allocated
     * in global memory.
     */

    global ctype *y     = shared_mem + (traj*9 + 0)*len;
    global ctype *dydt1 = shared_mem + (traj*9 + 1)*len;
    global ctype *dydt2 = shared_mem + (traj*9 + 2)*len;
    global ctype *dydt3 = shared_mem + (traj*9 + 3)*len;
    global ctype *dydt4 = shared_mem + (traj*9 + 4)*len;
    global ctype *dydt5 = shared_mem + (traj*9 + 5)*len;
    global ctype *dydt6 = shared_mem + (traj*9 + 6)*len;
    global ctype *dydt7 = shared_mem + (traj*9 + 7)*len;
    global ctype *ytmp  = shared_mem + (traj*9 + 8)*len;

    char error = 0;

    stype t = 0.0;

    // initialize psi
    for(idxtype i=0;i<len;i++){
      y[i] = psi0[i];
    }

    // compute length of y
    stype ylen2 = norm_sqr(y);

    // generate the first collaps time
    stype rand_vals_0 = random(&seed);

    stype h = hstart;

    // initialize output arrays to NaN
    status_out[traj] = -1;"""

    if len(e_ops) > 0:
        cl_src += r"""
    for(uint tidx=0;tidx<num_times;tidx++){
        for(uint j=0;j<e_ops_len;j++){
            expect_out[traj*e_ops_len*num_times + j*num_times + tidx] = NAN;
        }
    }"""

    if options.store_states:
        cl_src += r"""
    for(uint tidx=0;tidx<num_times;tidx++){
        for(idxtype i=0;i<len;i++){
            states_out[traj*num_times*len + tidx*len + i] = NAN;
        }
    }"""

    elif options.store_final_state:
        cl_src += r"""
    for(idxtype i=0;i<len;i++){
        states_out[traj*len + i] = NAN;
    }"""

    cl_src += r"""
    for(uint tidx=0;(tidx<num_times)&&(error==0);tidx++){

      int steps = 0;
      stype tend = tlist[tidx];

      while(true){

        if(++steps > nsteps){
          error = 1;
          break;
        }

        // compute runge kutta increments

        integrand(data,ind,ptr,t+c1*h,y,dydt1);

        for(idxtype i=0;i<len;i++){
          ytmp[i] = y[i]+h*a21*dydt1[i];
        }
        integrand(data,ind,ptr,t+c2*h,ytmp,dydt2);

        for(idxtype i=0;i<len;i++){
          ytmp[i] = y[i]+h*(a31*dydt1[i]+a32*dydt2[i]);
        }
        integrand(data,ind,ptr,t+c3*h,ytmp,dydt3);

        for(idxtype i=0;i<len;i++){
          ytmp[i] = y[i]+h*(a41*dydt1[i]+a42*dydt2[i]+a43*dydt3[i]);
        }
        integrand(data,ind,ptr,t+c4*h,ytmp,dydt4);

        for(idxtype i=0;i<len;i++){
          ytmp[i] = y[i]+h*(a51*dydt1[i]+a52*dydt2[i]+a53*dydt3[i]
                    +a54*dydt4[i]);
        }
        integrand(data,ind,ptr,t+c5*h,ytmp,dydt5);

        for(idxtype i=0;i<len;i++){
          ytmp[i] = y[i]+h*(a61*dydt1[i]+a62*dydt2[i]+a63*dydt3[i]
                    +a64*dydt4[i]+a65*dydt5[i]);
        }
        integrand(data,ind,ptr,t+c6*h,ytmp,dydt6);

        for(idxtype i=0;i<len;i++){
          ytmp[i] = y[i]+h*(a71*dydt1[i]+a72*dydt2[i]+a73*dydt3[i]
                    +a74*dydt4[i]+a75*dydt5[i]+a76*dydt6[i]);
        }
        integrand(data,ind,ptr,t+c7*h,ytmp,dydt7);

        // compute error
        stype error_quotient_sqr = 1e9;
        for(idxtype i=0;i<len;i++){
          const stype error2 = cabs2((b1-b1p)*dydt1[i]+(b2-b2p)*dydt2[i]
                   +(b3-b3p)*dydt3[i]+(b4-b4p)*dydt4[i]+(b5-b5p)*dydt5[i]
                   +(b6-b6p)*dydt6[i]+(b7-b7p)*dydt7[i]);
          //const stype tol2 = sqr( atol + rtol*length(y[i]) );
          error_quotient_sqr = min(atol2/error2, error_quotient_sqr);
        }

        // compute new h
        stype new_h = h * clamp( 0.84*rootn(error_quotient_sqr,10), 0.1, 4.0 );

        if(error_quotient_sqr >= 1.0){

            // compute next y in ytmp
            stype normsqr = 0.0f;
            for(idxtype i=0;i<len;i++){
                ytmp[i] = y[i] + h*(b1*dydt1[i]+b2*dydt2[i]+b3*dydt3[i]
                    +b4*dydt4[i]+b5*dydt5[i]+b6*dydt6[i]+b7*dydt7[i]);
                normsqr += sqr(ytmp[i].s0) + sqr(ytmp[i].s1);
            }

            // collaps
            if(normsqr <= rand_vals_0){

              if(rand_vals_0 - normsqr < norm_rtol*rand_vals_0){

                // advance time
                t += h;

                // set new y
                for(idxtype i=0;i<len;i++){
                  y[i] = ytmp[i];
                }
                ylen2 = normsqr;

                int j;
                stype sum_np = 0.0f;
                stype n_dp[c_ops_len];

                for(j=0;j<c_ops_len;j++){
                  constant const ptrtype *ptr2 = ptr+(len+1)*(H_len+j);
                  stype value = cy_expect_psi_csr(data,ind,ptr2,y).s0;"""

    for i, c_op in enumerate(c_ops):
        if isinstance(c_op, (list, tuple)):
            if isinstance(c_op[1], six.string_types):
                cl_src += "\n                  if(j==%d){value *= sqr(%s);}" % (i, c_op[1])
            else:
                cl_src += "\n                  if(j==%d){value *= sqr(%s)+sqr(%s);}" % (i, c_op[1][0], c_op[1][1])

    cl_src += r"""
                  n_dp[j] = value;
                  sum_np += value;
                }

                // select operand (j)
                stype rand_vals_1 = sum_np*random(&seed);
                sum_np = 0.0f;
                for(j=0; j<c_ops_len && sum_np<=rand_vals_1; j++){
                  sum_np += n_dp[j];
                }
                j = j - 1;

                for(idxtype i=0;i<len;i++){ytmp[i]=0.0;}
                ctype factor = (ctype)(1.0, 0.0);"""

    for i, c_op in enumerate(c_ops):
        if isinstance(c_op, (list, tuple)):
            if isinstance(c_op[1], six.string_types):
                cl_src += "\nif(j==%d){factor = (ctype)(%s,0.0);}" % (i, c_op[1])
            else:
                cl_src += "\nif(j==%d){factor = (ctype)(%s,%s);}" % (i, c_op[1][0], c_op[1][1])

    cl_src += r"""
                spmv_csr_add(data,ind,ptr+(len+1)*(H_len+c_ops_len+j),y,ytmp,factor);

                // compute psi_len_sqr again (after collaps op)
                stype psi_len_sqr = rsqrt(norm_sqr(ytmp));

                // norm vector
                for(idxtype i=0;i<len;i++){
                  y[i] = ytmp[i]*psi_len_sqr;
                }

                // generate new collaps time
                rand_vals_0 = random(&seed);

              }
              else if(h<=hmin){
                error = 3;
                break;
              }
              else{
                new_h = 0.5f * h;
              }

            }
            else{
              // advance time
              t += h;

              // set new y
              for(idxtype i=0;i<len;i++){
                y[i] = ytmp[i];
              }
              ylen2 = normsqr;
            }

        }
        else if(h<=hmin){
          error = 2;
          break;
        }

        if(t<tend){
          h = clamp(new_h, hmin, min(tend-t, hmax) );
        }
        else{
          h = clamp(new_h, hmin, hmax );
          break;
        }
      }"""

    if len(e_ops) > 0:
        cl_src += r"""
      // compute expectation values
      for(uint j=0;j<e_ops_len;j++){
        constant const ptrtype *ptr2 = ptr + (len+1)*(H_len+c_ops_len*2+j);
        const ctype value = cy_expect_psi_csr(data,ind,ptr2,y)/ylen2;
        expect_out[traj*e_ops_len*num_times + j*num_times + tidx] = value;
      }"""

    if options.store_states:
        cl_src += r"""
      // write out state vectors
      const stype norm = rsqrt(ylen2);
      for(idxtype i=0;i<len;i++){
        states_out[traj*num_times*len + tidx*len + i] = y[i]*norm;
      }"""

    cl_src += r"""
    }"""

    if not options.store_states and options.store_final_state:
        cl_src += r"""
    const stype norm = rsqrt(ylen2);
    for(idxtype i=0;i<len;i++){
        states_out[traj*len + i] = y[i]*norm;
    }"""

    cl_src += r"""
    status_out[traj] = error;
}"""

    # enable all optimizations
    compile_options = ['-cl-mad-enable', '-cl-denorms-are-zero',
                       '-cl-no-signed-zeros', '-cl-fast-relaxed-math']
    if not double_prec:
        compile_options.append('-cl-single-precision-constant')

    # build program
    queue = cl.CommandQueue(context)
    program = cl.Program(context, cl_src)
    starttime = time.time()
    try:
        program = program.build(options=' '.join(compile_options))
    except cl.RuntimeError as e:
        logger.error("----------- generated OpenCL code ------------")
        for no, line in enumerate(cl_src.splitlines()):
            logger.error("%3d " % (no + 1) + line)
        logger.error("----------- generated OpenCL code ------------")
        raise e
    logger.info("build time: %.2f sec" % (time.time() - starttime))

    logger.debug("size of opencl binary: %d bytes" % program.get_info(cl.program_info.BINARY_SIZES)[0])

    # print program info
    kernel = program.mcsolve
    for i, dev in enumerate(context.devices):
        wg_size = kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, dev)
        wg_compile = kernel.get_work_group_info(cl.kernel_work_group_info.COMPILE_WORK_GROUP_SIZE, dev)
        wg_multiple = kernel.get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, dev)
        local_usage = kernel.get_work_group_info(cl.kernel_work_group_info.LOCAL_MEM_SIZE, dev)
        private_usage = kernel.get_work_group_info(cl.kernel_work_group_info.PRIVATE_MEM_SIZE, dev)
        build_log = program.get_build_info(dev, cl.program_build_info.LOG)
        logger.debug("[dev=%d] name: %s" % (i, dev.name))
        logger.debug("[dev=%d] max work group size: %d of %d" % (i, wg_size, dev.max_work_group_size))
        logger.debug("[dev=%d] compiled work group size:" % i + repr(wg_compile))
        logger.debug("[dev=%d] recommended work-group multiple: %d" % (i, wg_multiple))
        logger.debug("[dev=%d] local mem used: %d of %d" % (i, local_usage, dev.local_mem_size))
        logger.debug("[dev=%d] private mem used: %d" % (i, private_usage))
        logger.debug("[dev=%d] global mem size: %d" % (i, dev.global_mem_size))
        logger.debug("[dev=%d] build log: %s" % (i, build_log))

    # run kernel
    kernel_args = [buf_seeds, buf_tlist, buf_psi0, buf_data, buf_ind, buf_ptr]
    if options.store_states or options.store_final_state:
        kernel_args.append(buf_states_out)
    if len(e_ops) > 0:
        kernel_args.append(buf_expect_out)
    kernel_args.append(buf_status_out)
    kernel_args.append(buf_shared_mem)
    global_size = (programcfg['WG_X'], programcfg['WG_Y'], programcfg['WG_Z'])
    starttime = time.time()
    complete_event = kernel(queue, global_size, None, *kernel_args)

    try:
        complete_event.wait()
        logger.info("execution time: %.2f sec" % (time.time() - starttime))
    except cl.RuntimeError as e:
        logger.error(
            'On NVIDIA platforms, the "out of resources" error can be raised '
            'if the device is also used for display, and the calculation '
            'takes longer than 5 seconds. It can also indicate a problem in '
            'pointer arithmetic.')
        raise e

    # return results
    if options.store_states or options.store_final_state:
        cl.enqueue_copy(queue, np_states_out, buf_states_out)
    if len(e_ops) > 0:
        cl.enqueue_copy(queue, np_expect_out, buf_expect_out)
    cl.enqueue_copy(queue, np_status_out, buf_status_out)

    # check status words
    fail = [0, 0, 0, 0]  # [success, nsteps, atol, norm_tol]
    for x in np_status_out:
        try:
            fail[x] += 1
        except:
            pass
    if sum(fail) < ntraj_int:
        logger.error("%d trajectories did not finish (unknown problem)" % fail[0])
    if fail[1] > 0:
        logger.warning("%d trajectories reached the maximum number of iterations" % fail[1])
    if fail[2] > 0:
        logger.warning("atol could not be reached for %d trajectories" % fail[2])
    if fail[3] > 0:
        logger.warning("norm_tol could not be reached for %d trajectories" % fail[3])

    # reshape output arrays
    if options.store_states:
        np_states_out.shape = (ntraj_int, num_times, dim, 1)
    elif options.store_final_state:
        np_states_out.shape = (ntraj_int, dim, 1)
    if len(e_ops) > 0:
        np_expect_out.shape = (ntraj_int, len(e_ops), num_times)

    # Store results in the Result object
    output = Result()
    output.solver = 'mcsolve_cl'
    output.seeds = seeds
    output.states = []

    starttime = time.time()

    # state vectors
    if options.store_states:
        if options.average_states:
            output.states = []
            for tidx in range(num_times):
                tmp = np.zeros((dim, dim), dtype=np.complex)
                for traj in range(ntraj_int):
                    state = np_states_out[traj][tidx]  # shape = (dim,1)
                    tmp += state * state.conj().transpose()
                # output.states.append(Qobj(tmp/tmp.trace()))
                output.states.append(Qobj(tmp / ntraj_int))

        else:
            output.states = []
            for traj in np_states_out:
                tmp = [Qobj(timepoint, dims=psi0.dims, shape=psi0.shape)
                       for timepoint in traj]
                output.states.append(tmp)

            # for compatibility with mcsolve
            if ntraj_int == 1:
                output.states = output.states[0]

    elif options.store_final_state:
        if options.steady_state_average:
            tmp = np.zeros((dim, dim), dtype=np.complex)
            for traj in range(ntraj_int):
                if np_status_out[traj] > 0:
                    state = np_states_out[traj]  # shape = (dim,1)
                    tmp += state * state.conj().transpose()
            output.states = [Qobj(tmp / fail[0])]
        else:
            output.states = []
            for traj in np_states_out:
                state = Qobj(traj, dims=psi0.dims, shape=psi0.shape)
                output.states.append(state)

    # for compatibility with mcsolve
    output.states = np.asarray(output.states, dtype=object)

    # expectation values
    if len(e_ops) > 0:
        if options.average_expect and ntraj_int > 1:
            # averaging if multiple trajectories
            if isinstance(ntraj, int):
                means = np.nanmean(np_expect_out, axis=0)
                output.expect = [means[e_op].real if e_ops[e_op].isherm else
                                 means[e_op] for e_op in range(len(e_ops))]
            elif isinstance(ntraj, (list, np.ndarray)):
                output.expect = []
                for num in ntraj:
                    means = np.nanmean(np_expect_out[:num], axis=0)
                    tmp = [means[e_op].real if e_ops[e_op].isherm else
                           means[e_op] for e_op in range(len(e_ops))]
                    output.expect.append(tmp)
        else:
            # no averaging for single trajectory or if average_expect flag
            # (Options) is off
            output.expect = []
            for traj in range(ntraj_int):
                a = [np_expect_out[traj][e_op].real if e_ops[e_op].isherm else
                     np_expect_out[traj][e_op] for e_op in range(len(e_ops))]
                output.expect.append(a)
            if ntraj_int == 1:
                output.expect = output.expect[0]

    logger.info("averaging time: %.2f sec" % (time.time() - starttime))

    # simulation parameters
    output.times = tlist
    output.num_expect = len(e_ops)
    output.num_collapse = len(c_ops)
    output.ntraj = ntraj_int

    if e_ops_dict:
        output.expect = {e: output.expect[n]
                         for n, e in enumerate(e_ops_dict.keys())}

    return output


def _check_operator_format(op):
    """Verifies that the argument has one of the following forms:
    * op
    * [op, 'real']
    * [op, ('real', 'imag')]
    """
    if isinstance(op, Qobj):
        pass
    else:
        if not isinstance(op, (list, tuple)):
            raise ValueError("invalid operator format, must be a Qobj or a list")
        if len(op) != 2:
            raise ValueError("invalid operator format, list must have two entries")
        if not isinstance(op[0], Qobj):
            raise ValueError("invalid operator format, first list item must be a Qobj")
        if isinstance(op[1], (list, tuple)):
            if len(op[1]) != 2:
                raise ValueError("invalid operator format, time dependence tuple must have two entries")
            if not isinstance(op[1][0], six.string_types) or not isinstance(op[1][1], six.string_types):
                raise ValueError("invalid operator format, time dependence tuple must contain a string")
        elif not isinstance(op[1], six.string_types):
            raise ValueError("invalid operator format, second list item must be a string or a tuple")


def _nptype2cl(nptype):
    "Returns the OpenCL equivalent of a numpy type or raises a ValueError."
    if nptype == np.uint8:
        return 'uchar'
    if nptype == np.uint16:
        return 'ushort'
    if nptype == np.uint32:
        return 'uint'
    if nptype == np.uint64:
        return 'ulong'
    if nptype == np.int8:
        return 'char'
    if nptype == np.int16:
        return 'short'
    if nptype == np.int32:
        return 'int'
    if nptype == np.int64:
        return 'long'
    if nptype == np.float16:
        return 'half'
    if nptype == np.float32:
        return 'float'
    if nptype == np.float64:
        return 'double'
    if nptype == np.complex64:
        return 'float2'
    if nptype == np.complex128:
        return 'double2'
    raise ValueError('no such opencl type: ' + nptype.__name__)


def list_opencl_platforms():
    """Print a list of available OpenCL platforms and their devices."""
    for n, platform in enumerate(cl.get_platforms()):
        print("platform", n)
        print("  name:", platform.name)
        print("  vendor:", platform.vendor)
        print("  version:", platform.version)
        for device in platform.get_devices():
            print("  device:")
            print("    name:", device.name)
            print("    compute units:", device.max_compute_units)
            print("    clock speed:", device.max_clock_frequency, "MHz")
            print("    memory size:",
                  device.global_mem_size // 1024 // 1024, "MB")
            print("    supports double precision:",
                  device.double_fp_config != 0)
