TARGET = qutraj_run

F90 = gfortran
F90FLAGS =
LDFLAGS = -lblas -llapack
F77 = gfortran
F77FLAGS =

PYFFLAGS = --overwrite-signature
F2PYBUILDFLAGS = -DF2PY_REPORT_ON_ARRAY_COPY=1

ZVODE = zvode/zvode zvode/zcopy zvode/zgbfa zvode/zgbsl zvode/zgefa zvode/zgesl
ZVODE_OBJS = $(ZVODE:=.o)
ZVODE_SRCS = $(ZVODE:=.f)

MODS = qutraj_precision
MODS += mt19937
MODS += linked_list
MODS += qutraj_general
MODS += qutraj_hilbert
MODS += qutraj_evolve

# Compile linear algebra routines? Depends on LAPACK

MODS += qutraj_linalg
#MODS += qutraj_nolinalg


OBJS = $(MODS:=.o)
SRCS += $(MODS:=.f90)

.PHONY: all clean

#all:
#	@echo Usage:
#	@echo make pyf - makes .pyf file, do this first.
#	@echo make f2pybuild - compiles the module and all dependencies.

all: $(OBJS)
	f2py $(F2PYBUILDFLAGS) -c $(TARGET:=.pyf) $(TARGET:=.f90) $(SRCS) $(ZVODE_OBJS) $(LDFLAGS)
	@echo
	@echo Remember to manually edit .pyf file to remove private subroutines and give default values to optional parameters. BEFORE building.

#$(OBJS): $(ZVODE_OBJS)

pyf:
ifneq "$(wildcard qutraj_run.pyf )" ""
	-cp $(TARGET:=.pyf) $(TARGET:=.pyf.backup)
endif
	f2py -m $(TARGET) -h $(TARGET:=.pyf) $(PYFFLAGS) $(TARGET:=.f90)
	@echo
	@echo Remember to manually edit .pyf file to remove private subroutines, give default values to optional parameters.

#
# need to manually edit .pyf file to remove private subroutines,
# and give default values for optional arguments
#

clean:
	$(RM) *.o
	$(RM) *.mod
	$(RM) *.so


.SUFFIXES: $(SUFFIXES) .f90 .f

# Need to override standard rule for mod files since make thinks they are modula-2 files
%.o: %.mod
.f90.o:
# Rule on how to convert from .f90 to .o
# $< macro refers to the name of the related file that caused the action.
	$(F90) $(F90FLAGS) -c $< 
.f90.mod:
	$(F90) $(F90FLAGS) -c $< 
.f.o:
	$(F77) $(F77FLAGS) -c -fPIC $< 

#
# Dependencies
#

qutraj_run: qutraj_run.f90 qutraj_precision.o qutraj_general.o qutraj_hilbert.o qutraj_linalg.o qutraj_evolve.o linked_list.o mt19937.o
qutraj_evolve: qutraj_evolve.f90 qutraj_precision.o qutraj_general.o qutraj_hilbert.o linked_list.o mt19937.o zvode/zvode.o
qutraj_linalg: qutraj_linalg.f90 qutraj_hilbert.o qutraj_precision.o qutraj_general.o
qutraj_hilbert: qutraj_hilbert.f90 qutraj_precision.o qutraj_general.o
qutraj_general: qutraj_general.f90 qutraj_precision.o
linked_list: linked_list.f90 qutraj_precision.o
mt19937: mt19937.f90 qutraj_precision.o
qutraj_precision: qutraj_precision.f90

