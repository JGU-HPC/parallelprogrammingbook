UPCXXHOME= /opt/upcxx/
UPCXX= $(UPCXXHOME)/bin/upc++
UPCXXINC= $(UPCXXHOME)/include/upcxx/
UPCXXFLAGS= -O2 -std=c++11 -DGASNET_SEQ -DUSE_GASNET_FAST_SEGMENT -DONLY_MSPACES
GASNETRUN= /opt/gasnet/bin/gasnetrun_mpi -n 4 # install gasnet and choose backend

all: matrix_vector

matrix_vector: matrix_vector.cxx
	$(UPCXX) $(UPCXXFLAGS) matrix_vector.cxx -o matrix_vector -I $(UPCXXINC)

matrix_vector_run: matrix_vector
	$(GASNETRUN) matrix_vector 128 256


clean:
	rm -rf matrix_vector
