# Generate src/intdtype.h into the *source* tree. It has to be there, not just
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

# -- Include .py, .pxd, .pxi, .pyi, .ini ---
py3.install_sources(
  [
    '__init__.py', 'about.py', 'animation.py', 'bloch.py', 'cite.py',
    'continuous_variables.py', 'control.py', 'distributions.py', 'entropy.py',
    'fileio.py', 'ipynbtools.py', 'matplotlib_utilities.py', 'measurement.py',
    'partial_transpose.py', 'qip.py', 'random_objects.py', 'settings.py',
    'simdiag.py', 'testing.py', 'tomography.py', 'typing.py', 'utilities.py',
    'visualization.py', 'wigner.py',
  ],
  subdir: 'qutip',
)

foreach pkg : ['core', 'solver', 'ui', '_mkl', 'legacy', 'piqs', 'tests']
  install_subdir(
    pkg,
    install_dir: py3.get_install_dir() / 'qutip',
    exclude_directories: ['__pycache__'],
  )
endforeach

# ---- extension modules ----
cy_modules = {
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
  'solver/integrator': ['explicit_rk'],
  'solver/sode':       ['_sode', 'ssystem'],
}

foreach reldir, names : cy_modules
  foreach name : names
    py3.extension_module(
      name,
      reldir / name + '.pyx',
      dependencies: [qutip_dep],
      install: true,
      subdir: 'qutip' / reldir,
    )
  endforeach
endforeach

py3.extension_module(
  '_distributions', '_distributions.pyx',
  dependencies: [qutip_dep], install: true, subdir: 'qutip',
)

# ---- these two need more than the default ----
py3.extension_module(
  'matmul',
  [
    'core/data/matmul.pyx',
    'core/data/src/matmul_csr_vector.cpp',
    'core/data/src/matmul_csr_dense.cpp',
    'core/data/src/matmul_diag_vector.cpp',
  ],
  dependencies: [qutip_dep],
  install: true, subdir: 'qutip/core/data',
)