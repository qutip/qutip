from setuptools import setup, find_packages
 
execfile('qutip/_version.py')
setup(
    name = "QuTIP",
    version = __version__,
    packages = find_packages(),
    author = "Paul D. Nation, Robert J. Johansson",
    author_email = "pnation@riken.jp, robert@riken.jp",
    license = "GPL3",
    url = "http://dml.riken.jp"
    
)
