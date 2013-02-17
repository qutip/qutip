TARGET = zvode

F90 = gfortran
F90FLAGS =
LDFLAGS = -lblas
F77 = gfortran
F77FLAGS =

ZVODE = zvode zcopy zgbfa zgbsl zgefa zgesl
ZVODE_OBJS = $(ZVODE:=.o)
ZVODE_SRCS = $(ZVODE:=.f)

.PHONY: all clean

all: $(ZVODE_OBJS)

clean:
	$(RM) *.o
	$(RM) *.mod
	$(RM) *.so

.SUFFIXES: $(SUFFIXES) .f90 .f
	#
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


