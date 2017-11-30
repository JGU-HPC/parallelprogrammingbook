UPCXXHOME= /opt/upcxx/
UPCXX= $(UPCXXHOME)/bin/upc++
UPCXXINC= $(UPCXXHOME)/include/upcxx/
UPCXXFLAGS= -O2 -std=c++11 -DGASNET_SEQ -DUSE_GASNET_FAST_SEGMENT -DONLY_MSPACES
GASNETRUN= /opt/gasnet/bin/gasnetrun_mpi -n 4 # install gasnet and choose backend

all: mandel1 mandel2

mandel1: mandel1.cxx
	$(UPCXX) $(UPCXXFLAGS) mandel1.cxx -o mandel1 -I $(UPCXXINC)

mandel1_run: mandel1
	$(GASNETRUN) mandel1 512 512 1024

mandel2: mandel2.cxx
	$(UPCXX) $(UPCXXFLAGS) mandel2.cxx -o mandel2 -I $(UPCXXINC)

mandel2_run: mandel2
	$(GASNETRUN) mandel2 512 512 1024

clean:
	rm -rf mandel1
	rm -rf mandel2
