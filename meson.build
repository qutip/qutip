project(
  'qutip',
  'c', 'cpp', 'cython',
  version: files('VERSION'),
  license: 'BSD-3',
  meson_version: '>= 1.5.0',
  default_options: [
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
    'c_std=c17',
    'cpp_std=c++17',
    'blas=openblas',
    'lapack=openblas'
  ],
)

fs = import('fs')

py3 = import('python').find_installation(pure: false)
py3_dep = py3.dependency()

# Emit a warning for 32-bit Python installs on Windows; users are getting
# unexpected from-source builds there because we no longer provide wheels.
is_windows = host_machine.system() == 'windows'
if is_windows and py3.has_variable('EXT_SUFFIX')
  ext_suffix = py3.get_variable('EXT_SUFFIX')
  if ext_suffix.contains('win32')
    warning('You are building from source on a 32-bit Python install. QuTiP does not provide 32-bit wheels; install 64-bit Python if you are having issues!')
  endif
endif

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')
cy = meson.get_compiler('cython')
# generator() doesn't accept compilers, only found programs - cast it.
cython = find_program(cy.cmd_array()[0])

if not cc.has_header('Python.h', dependencies: py3_dep)
  error('Cannot compile Python.h')
endif

# Check compiler is recent enough (see "Toolchain Roadmap" for details)
if cc.get_id() == 'gcc'
  if not cc.version().version_compare('>=10.3')
    error('QuTiP requires GCC >= 10.3')
  endif
elif cc.get_id() == 'clang' or cc.get_id() == 'clang-cl'
  if not cc.version().version_compare('>=15.0')
    error('QuTiP requires clang >= 15.0')
  endif
elif cc.get_id() == 'msvc'
  if not cc.version().version_compare('>=19.20')
    error('QuTiP requires at least vc142 (default with Visual Studio 2019) ' + \
          'when building with MSVC')
  endif
endif

_global_c_args = cc.get_supported_arguments(
  '-Wno-unused-but-set-variable',
  '-Wno-unused-function',
  '-Wno-conversion',
  '-Wno-misleading-indentation',
)
add_project_arguments(_global_c_args, language : 'c')

# We need -lm for all C code (assuming it uses math functions, which is safe to
# assume for QuTiP). For C++ it isn't needed, because libstdc++/libc++ is
# guaranteed to depend on it.
m_dep = cc.find_library('m', required : false)
if m_dep.found()
  add_project_link_arguments('-lm', language : 'c')
endif

if host_machine.system() == 'os400'
  # IBM i system, needed to avoid build errors - see gh-17193
  add_project_arguments('-D__STDC_FORMAT_MACROS', language : 'cpp')
  add_project_link_arguments('-Wl,-bnotextro', language : ['c', 'cpp'])
endif

if cc.get_id() == 'clang-cl'
  add_global_arguments('-D_CRT_SECURE_NO_WARNINGS', language: ['c', 'cpp'])
  # Some arguments like -Wl,--version-script may otherwise generate a
  # large number of warnings. clang-cl accepts but ignores them.
  add_project_arguments('-Qunused-arguments', language: ['c', 'cpp'])
endif

if cpp.get_id() == 'msvc'
  add_project_arguments('/bigobj', language: 'cpp')
endif

# Silence GCC -Wpsabi notes about ABI changes (e.g., the GCC 10.1 ABI changes
# for passing parameter passing for argument of type ‘std::pair<double, double>`).
# These notes are informational; SciPy is built against a single compiler ABI.
if meson.get_compiler('cpp').get_id() == 'gcc'
  # global rather than project-wide, because it silences a few warnings from
  # within subprojects/highs as well
  add_global_arguments('-Wno-psabi', language: 'cpp')
endif

if host_machine.system() == 'darwin'
  if cc.has_link_argument('-Wl,-dead_strip')
    # Allow linker to strip unused symbols
    add_project_link_arguments('-Wl,-dead_strip', language : ['c', 'cpp'])
  endif
endif

if get_option('b_sanitize') != 'none'
  if cc.get_id() == 'clang'
    ignore_file = meson.current_source_dir() / 'tools' / 'asan-ignore.txt'
    add_global_arguments('-fsanitize-ignorelist=' + ignore_file, language: ['c', 'cpp'])
  endif
endif

# Intel compilers default to fast-math, so disable it if we detect Intel
# compilers. A word of warning: this may not work with the conda-forge
# compilers, because those have the annoying habit of including lots of flags
# that are gcc-specific in CFLAGS/CXXFLAGS/FFLAGS, which throws off the
# detection logic below. You have to remove the wrong flags (only `-isystem`
# is actually needed, everything else shouldn't be there).
_intel_cflags = []
if cc.get_id() in ['intel', 'intel-llvm']
  _intel_cflags += cc.get_supported_arguments('-fp-model=strict')
elif cc.get_id() in ['intel-cl', 'intel-llvm-cl']
  _intel_cflags += cc.get_supported_arguments('/fp:strict')
endif
add_global_arguments(_intel_cflags, language: ['c', 'cpp'])

# Hide symbols when building on Linux with GCC. For Python extension modules,
# we only need `PyInit_*` to be public, anything else may cause problems. So we
# use a linker script to avoid exporting those symbols (this is in addition to
# Meson using `-fvisibility=hidden` for C and `-fvisibility-inlines-hidden` for
# C++ code. See gh-15996 for details.
version_link_args = []
if cc.get_id() != 'clang-cl'
  _linker_script = meson.project_source_root() / 'tools/building/link-version-pyinit.map'
  vscript_link_args = ['-Wl,--version-script=' + _linker_script]
  # Note that FreeBSD only accepts version scripts when -shared is passed,
  # hence we need to pass that to `cc.links` explicitly (flag is already
  # present for `extension_module` invocations).
  if cc.links('', name: '-Wl,--version-script', args: ['-shared', vscript_link_args])
    version_link_args = vscript_link_args
  endif
endif

subdir('qutip')