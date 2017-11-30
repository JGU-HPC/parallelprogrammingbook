UPCXXHOME= /opt/upcxx/
UPCXX= $(UPCXXHOME)/bin/upc++
UPCXXINC= $(UPCXXHOME)/include/upcxx/
UPCXXFLAGS= -O2 -std=c++11 -DGASNET_SEQ -DUSE_GASNET_FAST_SEGMENT -DONLY_MSPACES
GASNETRUN= /opt/gasnet/bin/gasnetrun_mpi -n 4 # install gasnet and choose backend

all: letter1 letter2

letter1: letter1.cxx
	$(UPCXX) $(UPCXXFLAGS) letter1.cxx -o letter1 -I $(UPCXXINC)

letter1_run: letter1
	$(GASNETRUN) letter1 C 4

letter2: letter2.cxx
	$(UPCXX) $(UPCXXFLAGS) letter2.cxx -o letter2 -I $(UPCXXINC)

letter2_run: letter2
	$(GASNETRUN) letter2 C 4

clean:
	rm -rf letter1
	rm -rf letter2
