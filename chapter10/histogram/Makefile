UPCXX= /opt/upcxx/bin/upc++                   # install UPC++ and specify binary
UPCXXFLAGS= -O2 -std=c++11 -DGASNET_SEQ -DUSE_GASNET_FAST_SEGMENT -DONLY_MSPACES
GASNETRUN= /opt/gasnet/bin/gasnetrun_mpi -n 4 # install gasnet and choose backend

all: histo1 histo2 histo3

histo1: histo1.cxx
	$(UPCXX) $(UPCXXFLAGS) histo1.cxx -o histo1

histo1_run: histo1
	$(GASNETRUN) histo1 16 16

histo2: histo2.cxx
	$(UPCXX) $(UPCXXFLAGS) histo2.cxx -o histo2

histo2_run: histo2
	$(GASNETRUN) histo2 16 16

histo3: histo3.cxx
	$(UPCXX) $(UPCXXFLAGS) histo3.cxx -o histo3

histo3_run: histo3
	$(GASNETRUN) histo3 16 16

clean:
	rm -rf histo1
	rm -rf histo2
	rm -rf histo3
