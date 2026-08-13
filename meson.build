project(
  'qutip',
  'cpp', 'cython',
  version: run_command(['tools/git-versioning.py', '--write'], check: true).stdout().strip(),
  license: 'BSD-3-Clause',
  meson_version: '>= 1.5.0',
  default_options: [
    'buildtype=release',
    'cpp_std=c++17',
    'cython_language=cpp',  # all our .pyx are C++, avoids per-target override
  ],
)

py3 = import('python').find_installation(pure: false)
py3_dep = py3.dependency()

cpp = meson.get_compiler('cpp')
if not cpp.has_header('Python.h', dependencies: py3_dep)
  error('Cannot compile Python.h')
endif

# compile flags for C++ on top of -O3/-O2 (used in release)
_extra_args = [
  '-funroll-loops',
  '-Wno-unused-but-set-variable',
  '-Wno-unused-function',
  '-Wno-conversion',
  '-Wno-misleading-indentation',
]
add_project_arguments(cpp.get_supported_arguments(_extra_args), language: 'cpp')

# ---- Cython annotate ----
if get_option('annotate')
  add_project_arguments('-a', language: 'cython')
endif

# ---- 32-bit Windows warning ----
if host_machine.system() == 'windows' and py3.has_variable('EXT_SUFFIX')
  if py3.get_variable('EXT_SUFFIX').contains('win32')
    warning('Building from source on 32-bit Python; QuTiP ships no 32-bit wheels.')
  endif
endif

# ---- numpy ----
incdir_numpy = run_command(
  py3, ['-c', 'import numpy; print(numpy.get_include())'], check: true,
).stdout().strip()
np_dep = declare_dependency(
  include_directories: include_directories(incdir_numpy),
  # --  compile_args: ['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'],
)

# ---- warning suppression, applied to C++ ----
_warn_args = [
  '-Wno-unused-but-set-variable',
  '-Wno-unused-function',
  '-Wno-conversion',
  '-Wno-misleading-indentation',
]
add_project_arguments(cpp.get_supported_arguments(_warn_args), language: 'cpp')

if cpp.get_id() == 'msvc'
  add_project_arguments('/bigobj', language: 'cpp')
elif cpp.get_id() == 'gcc'
  add_project_arguments('-Wno-psabi', language: 'cpp')
elif cpp.get_id() == 'clang-cl'
  add_project_arguments('-D_CRT_SECURE_NO_WARNINGS', '-Qunused-arguments',
                        language: 'cpp')
endif

if host_machine.system() == 'darwin' and cpp.has_link_argument('-Wl,-dead_strip')
  add_project_link_arguments('-Wl,-dead_strip', language: 'cpp')
endif

subdir('qutip')
install_subdir(
  'qutip',
  install_dir: py3.get_install_dir(),
  exclude_files: ['meson.build'],
  exclude_directories: ['__pycache__', 'tests/.pytest_cache'],
)