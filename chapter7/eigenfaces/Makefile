NVCC=nvcc
NVCCFLAGS=-O3 -std=c++11 -D_FORCE_INLINES -arch=sm_61

all: mean_computation                        \
     mean_correction_coalesced               \
     mean_correction_non_coalesced           \
     covariance_naive                        \
	 covariance_symmetric                    \
	 covariance_shared                       \
	 eigenfaces

mean_computation: mean_computation.cu
	$(NVCC) $(NVCCFLAGS) mean_computation.cu -o mean_computation

mean_correction_coalesced: mean_correction.cu
	$(NVCC) $(NVCCFLAGS) mean_correction.cu -o mean_correction_coalesced -DCOALESCED_ACCESS

mean_correction_non_coalesced: mean_correction.cu
	$(NVCC) $(NVCCFLAGS) mean_correction.cu -o mean_correction_non_coalesced

covariance_naive: covariance.cu
	$(NVCC) $(NVCCFLAGS) covariance.cu -o covariance_naive -DCOV_MODE_NAIVE

covariance_symmetric: covariance.cu
	$(NVCC) $(NVCCFLAGS) covariance.cu -o covariance_symmetric -DCOV_MODE_SYMMETRIC

covariance_shared: covariance.cu
	$(NVCC) $(NVCCFLAGS) covariance.cu -o covariance_shared

eigenfaces: eigenfaces.cu
	$(NVCC) $(NVCCFLAGS) eigenfaces.cu -o eigenfaces -lcusolver

clean:
	rm -f mean_computation                 \
	      mean_correction_coalesced        \
	      mean_correction_non_coalesced    \
 	      covariance_naive                 \
	      covariance_symmetric             \
	      covariance_shared                \
	      eigenfaces
