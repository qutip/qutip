#!/bin/sh

hash brew &> /dev/null
if [ $? -eq 1 ]; then
    echo 'Installing Homebrew ...'
    ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
fi

# Ensure Homebrew formulae are updated
brew update

hash git &> /dev/null
if [ $? -eq 1 ]; then
    echo 'Installing Git ...'
    brew install git
fi

hash gcc &> /dev/null
if [ $? -eq 1 ]; then
    echo 'No gcc detected; Installing XCode Command Line Tools ...'
    xcode-select --install
fi

# Add science tap
brew tap homebrew/science

# Do the brews
brew install python3
brew install gcc # This includes gfortran
brew install zeromq
brew install freetype
# OpenBLAS for NumPy/SciPy
#brew install openblas
#export BLAS=/usr/local/opt/openblas/lib/libopenblas.a
#export LAPACK=/usr/local/opt/openblas/lib/libopenblas.a

# General
pip3 install -U nose
pip3 install -U six
pip3 install -U patsy
pip3 install -U pygments
pip3 install -U sphinx
pip3 install -U cython
# IPython
pip3 install -U jinja2
pip3 install -U tornado
pip3 install -U pyzmq
pip3 install -U jsonschema
pip3 install -U ipython
# NumPy
pip3 install -U numpy
# SciPy
pip3 install -U scipy
# Matplotlib
pip3 install -U matplotlib
# QuTiP
pip3 install -U https://github.com/qutip/qutip/archive/master.tar.gz --install-option=--with-f90mc
