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
sudo pip3 install -U nose
sudo pip3 install -U six
sudo pip3 install -U patsy
sudo pip3 install -U pygments
sudo pip3 install -U sphinx
sudo pip3 install -U cython
# IPython
sudo pip3 install -U jinja2
sudo pip3 install -U tornado
sudo pip3 install -U pyzmq
sudo pip3 install -U jsonschema
sudo pip3 install -U ipython
# NumPy
sudo pip3 install -U numpy
# SciPy
sudo pip3 install -U scipy
# Matplotlib
sudo pip3 install -U matplotlib
# QuTiP
sudo pip3 install -U https://github.com/qutip/qutip/archive/master.tar.gz --install-option=--with-f90mc
