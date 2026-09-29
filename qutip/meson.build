# Generate src/intdtype.h into the source tree. It has to be there, not just
# in the build dir, because qutip/core/coefficient.py compiles string
# coefficients at runtime with its own setuptools invocation.

run_command(
  py3, '-c',
  '''
import pathlib, sys
size, target = sys.argv[1], pathlib.Path(sys.argv[2])
content = "typedef int%s_t idxint;\nstatic const int _idxint_size = %s;\n" % (size, size)
if not target.exists() or target.read_text() != content:
    target.write_text(content)
''',
  get_option('idxint_64') ? '64' : '32',
  meson.current_source_dir() / 'core' / 'data' / 'src' / 'intdtype.h',
  check: true,
)

qutip_dep = declare_dependency(
  dependencies: [np_dep],
  include_directories: include_directories('core/data'),
)

# ---- SciPy BLAS integer width ----
# SciPy >= 1.18 exports `blas_int` from cython_blas / cython_lapack: it is C
# `int` for a normal build and `int64_t` for one against an ILP64 BLAS. Probe
# for the name and generate core/data/_blas_int.pxi accordingly, so our .pyx
# files can `include "_blas_int.pxi"` and declare BLAS arguments as blas_int.
# Pattern from scipy/scipy#25025.
cy = meson.get_compiler('cython')
_blas_int_conf = configuration_data()
if cy.compiles('from scipy.linalg.cython_blas cimport blas_int',
               name: 'scipy cython_blas exports blas_int')
  _blas_int_conf.set('BLAS_INT_DEF',
                     'from scipy.linalg.cython_blas cimport blas_int')
else
  # SciPy < 1.18: always LP64.
  _blas_int_conf.set('BLAS_INT_DEF', 'ctypedef int blas_int')
endif

blas_int_pxi = configure_file(
  input: 'core/data/_blas_int.pxi.in',
  output: '_blas_int.pxi',
  configuration: _blas_int_conf,
)

# `include "_blas_int.pxi"` resolves against the Cython include path.
add_project_arguments('-I' + meson.current_build_dir(), language: 'cython')

# ---- extension modules ----
cy_modules = {
  '.':                 ['_distributions'],
  'core':              ['_brtools', '_brtensor'],
  'core/cy':           ['_element', 'coefficient', 'lindblad_matrix_form',
                        'math', 'qobjevo'],
  'core/data':         ['add', 'adjoint', 'base', 'block_operations',
                        'convert', 'csr', 'data_iterator', 'dense', 'dia',
                        'dispatch', 'expect', 'inner', 'kron', '_local_matmul',
                        'mean', 'mul', 'norm', 'ode', 'permute', 'pow',
                        'project', 'properties', 'ptrace', 'reshape', 'tidyup',
                        'trace'],
  'piqs':              ['_piqs'],
  'solver/cy':         ['dysolve', 'nm_mcsolve'],
  'solver/integrator': ['explicit_rk', '_rhs'],
  'solver/sode':       ['_sode', 'ssystem'],
}

foreach reldir, names : cy_modules
  foreach name : names
    py3.extension_module(
      name,
      [reldir / name + '.pyx', blas_int_pxi],
      dependencies: [qutip_dep],
      install: true,
      subdir: 'qutip' / reldir,
    )
  endforeach
endforeach

# ---- explicitly compile .cpp files ----
py3.extension_module(
  'matmul',
  [
    'core/data/matmul.pyx',
    blas_int_pxi,
    'core/data/src/matmul_csr_vector.cpp',
    'core/data/src/matmul_csr_dense.cpp',
    'core/data/src/matmul_diag_vector.cpp',
  ],
  dependencies: [qutip_dep],
  install: true, subdir: 'qutip/core/data',
)