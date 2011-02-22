#COMP	= mpicc

# on beach
# COMP = /usr/local/mvapich2/bin/mpicc
# COMP = /usr/local/mpich2-1-1-1p1/bin/mpicc
COMP = /usr/local/mpich2-1.1/bin/mpicc
# COMP   = /usr/local/mpich-2.1.0.8/bin/mpicc
# COMP  = /usr/local/openmpi-intel/bin/mpicc
# COMP = /usr/local/openmpi-new/bin/mpicc
# COMP = mpicc

# Triton
# COMP    = /home/ariel/mahidhar/mpich-pgimx/bin/mpicc

OBJECTS = pllcart3dwp2.o pll_tridiag.o
FFLGS	= -lm #-v #-qcpluscmt #-O3 

# on caja
# LIBS	= -L/usr/local/lib -lfftw3 -lm -L/usr/lib64/mpich2 -lmpich -lpthread -lrt #-lmpichcxx
# on rafael-laptop
# LIBS	= -L/usr/local/lib -lfftw3 -lm -L/usr/local/lib -lmpich -lpthread -lrt -lmpichcxx
# on Dell
# LIBS	= -L/usr/local/lib -lfftw3 -lm -L/usr/local/mpich/lib -lmpich -lpthread -lrt
# on Teragrid
# LIBS	= -L/usr/local/apps/fftw301d/lib -lfftw3 -lm -L/usr/local/apps/mpich-gm-1.2.6..14b-intel-r2/lib -lmpich -lpthread -lrt
# on Triton
# LIBS  = -L/opt/pgi/fftw_pgi/lib -lfftw3
# on Ranger
# LIBS  = -L/share/apps/pgi7_2/fftw3/3.1.2/lib -lfftw3 -lm -L/share/apps/pgi7_2/mvapich/1.0.1/lib -lmpich -lpthread -lrt

# on beach
# LIBS = -L/usr/local/fftw-3.2.2/lib -lfftw3 -lm -L/usr/local/mvapich2/lib -lmpich -lpthread -lrt
# LIBS = -L/usr/local/fftw-3.2.2/lib -lfftw3 -lm -L/usr/local/mpich2-1-1-1p1/lib -lmpich -lpthread -lrt
# LIBS = -L/usr/local/fftw-3.2.2/lib -lfftw3 -lm -L/usr/local/mpich2-1.1/lib -lmpich -lpthread -lrt
  LIBS = -L/usr/local/fftw-3.2.2/lib -lfftw3 -lm -L/usr/local/mpich2-gfort-local/lib -lmpich -lpthread -lrt
# LIBS = -L/usr/local/fftw-3.2.2/lib -lfftw3 -lm -L/usr/local/openmpi-intel/lib -lmpi -lpthread -lrt
# for openmpi with module load
# LIBS = -L/usr/local/fftw-3.2.2/lib -lfftw3 -lm -lmpi -lpthread -lrt -L/opt/torque/lib
# LIBS = -L/usr/local/fftw-3.2.2/lib -lfftw3 -lm -L/usr/local/openmpi-new/lib -lmpi -lpthread -lrt -L/opt/torque/lib -ltorque
# LIBS = -L/usr/local/fftw-3.2.2/lib -lfftw3 -lm -L/usr/local/mpich-2.1.0.8/lib -lmpich -lpthread -lrt


run3.x: $(OBJECTS) 
	$(COMP) $(FFLGS) $(OBJECTS) $(LIBS) -o run3.x 
pllcart3dwp2.o: pllcart3d.c
	$(COMP) -c $(FFLGS) pllcart3d.c
pll_tridiag.o: pll_tridiag.c
	$(COMP) -c $(FFLGS) pll_tridiag.c
clean:
	rm *.o ; rm *.bin ; rm *.dat ; rm Outscreen ; rm *.x ; rm middata ; rm midfield ; rm *.png ; rm mvasubscript.*
