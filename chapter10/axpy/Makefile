UPCXXHOME= /opt/upcxx/
UPCXX= $(UPCXXHOME)/bin/upc++
UPCXXINC= $(UPCXXHOME)/include/upcxx/
UPCXXFLAGS= -O2 -std=c++11 -DGASNET_SEQ -DUSE_GASNET_FAST_SEGMENT -DONLY_MSPACES
GASNETRUN= /opt/gasnet/bin/gasnetrun_mpi -n 4 # install gasnet and choose backend

all: axpy

axpy: axpy.cxx
	$(UPCXX) $(UPCXXFLAGS) axpy.cxx -o axpy -I $(UPCXXINC)

axpy_run: axpy
	$(GASNETRUN) axpy 4 1

clean:
	rm -rf axpy
