from setuptools import setup, find_packages
 
execfile('qutip/_version.py')
setup(
    name = "QuTiP",
    version = __version__,
    packages = find_packages(),
    author = "Paul D. Nation, Robert J. Johansson",
    author_email = "pnation@riken.jp, robert@riken.jp",
    license = "GPL3",
    description = ("A framework desgined to solve for the dynamics of open-quantum systems."),
    keywords = "quantum physics dynamics",
    url = "http://code.google.com/p/qutip/",
    classifiers=[
            "Development Status :: 3 - Alpha",
            "Topic :: Science",
            "License :: GNU GPL3 License",
        ],
    include_package_data=True
    
    
)
