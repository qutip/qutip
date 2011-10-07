#
# used for building debian package
#
PYTHON=`which python`
DESTDIR=/
BUILDIR=$(CURDIR)/debian/qutip
PROJECT=qutip
VERSION=1.1.0

all:
	@echo "make source - Create source package"
	@echo "make install - Install on local system"
	@echo "make buildrpm - Generate a rpm package"
	@echo "make builddeb - Generate a deb package"
	@echo "make clean - Get rid of scratch and byte files"

source:
	$(PYTHON) setup.py sdist $(COMPILE)

install:
	$(PYTHON) setup.py install --root $(DESTDIR) $(COMPILE)

buildrpm:
	$(PYTHON) setup.py bdist_rpm --post-install=rpm/postinstall --pre-uninstall=rpm/preuninstall

builddeb:
	$(PYTHON) setup.py sdist $(COMPILE) --dist-dir=../ --prune
	rename -f 's/QuTiP-(.*)\.tar\.gz/qutip_$$1\.orig\.tar\.gz/' ../*
	dpkg-buildpackage -i -I -rfakeroot

clean:
	$(PYTHON) setup.py clean
	rm -rf build/ MANIFEST
	find . -name '*.pyc' -delete
