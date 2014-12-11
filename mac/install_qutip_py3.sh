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

# General
/usr/local/bin/pip3 install -U nose
/usr/local/bin/pip3 install -U six
/usr/local/bin/pip3 install -U patsy
/usr/local/bin/pip3 install -U pygments
/usr/local/bin/pip3 install -U sphinx
/usr/local/bin/pip3 install -U cython
# IPython
/usr/local/bin/pip3 install -U jinja2
/usr/local/bin/pip3 install -U tornado
/usr/local/bin/pip3 install -U pyzmq
/usr/local/bin/pip3 install -U jsonschema
/usr/local/bin/pip3 install -U ipython
# NumPy
/usr/local/bin/pip3 install -U numpy
# SciPy
/usr/local/bin/pip3 install -U scipy
# Matplotlib
/usr/local/bin/pip3 install -U matplotlib
# QuTiP
/usr/local/bin/pip3 install -U qutip --install-option=--with-f90mc

# run QuTiP tests from shell
echo "Running QuTiP unit tests"
/usr/local/bin/python3 -c "import qutip.testing as qt; qt.run()"

# check brew installation
brew doctor
