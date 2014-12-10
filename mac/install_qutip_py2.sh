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
brew install python
brew install gcc # This includes gfortran
brew install zeromq
brew install freetype

# General
/usr/local/bin/pip install -U nose
/usr/local/bin/pip install -U six
/usr/local/bin/pip install -U patsy
/usr/local/bin/pip install -U pygments
/usr/local/bin/pip install -U sphinx
/usr/local/bin/pip install -U cython
# IPython
/usr/local/bin/pip install -U jinja2
/usr/local/bin/pip install -U tornado
/usr/local/bin/pip install -U pyzmq
/usr/local/bin/pip install -U jsonschema
/usr/local/bin/pip install -U ipython
# NumPy
/usr/local/bin/pip install -U numpy
# SciPy
/usr/local/bin/pip install -U scipy
# Matplotlib
/usr/local/bin/pip install -U matplotlib
# QuTiP
/usr/local/bin/pip install -U qutip --install-option=--with-f90mc

# run QuTiP tests from shell
echo "Running QuTiP unit tests"
/usr/local/bin/python -c "import qutip.testing as qt; qt.run()"

# check brew installation
brew doctor
