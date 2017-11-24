CXX=g++
CXXFLAGS=-O2 -std=c++14 -Wall -fopenmp
AVXFLAGS=-mavx
NOVECTOR=

all: vector_norm_build vector_max_build matrix_matrix_mult_build pointwise_vector_max_build

vector_norm_soa_plain: vector_norm_soa_plain.cpp
	$(CXX) $(CXXFLAGS) vector_norm_soa_plain.cpp -o vector_norm_soa_plain $(NOVECTOR)

vector_norm_soa_avx: vector_norm_soa_avx.cpp
	$(CXX) $(CXXFLAGS) vector_norm_soa_avx.cpp -o vector_norm_soa_avx $(AVXFLAGS)

vector_norm_aos_plain: vector_norm_aos_plain.cpp
	$(CXX) $(CXXFLAGS) vector_norm_aos_plain.cpp -o vector_norm_aos_plain $(NOVECTOR)

vector_norm_aos_avx: vector_norm_aos_avx.cpp
	$(CXX) $(CXXFLAGS) vector_norm_aos_avx.cpp -o vector_norm_aos_avx $(AVXFLAGS)

vector_norm_build: vector_norm_soa_plain vector_norm_soa_avx \
	               vector_norm_aos_plain vector_norm_aos_avx

vector_norm_run: vector_norm_build
	@echo "#####################################"
	@echo "# Vector Normalization Benchmark"
	@echo "#####################################"
	./vector_norm_soa_plain
	./vector_norm_soa_avx
	./vector_norm_aos_plain
	./vector_norm_aos_avx

vector_max: vector_max.cpp
	$(CXX) $(CXXFLAGS) vector_max.cpp -o vector_max $(AVXFLAGS)

vector_max_build: vector_max

vector_max_run: vector_max_build
	@echo "#####################################"
	@echo "# Vector Reduction Benchmark"
	@echo "#####################################"
	./vector_max

matrix_matrix_mult: matrix_matrix_mult.cpp
	$(CXX) $(CXXFLAGS) matrix_matrix_mult.cpp -o matrix_matrix_mult $(AVXFLAGS)

matrix_matrix_mult_build: matrix_matrix_mult

matrix_matrix_mult_run: matrix_matrix_mult_build
	@echo "#####################################"
	@echo "# Matrix Matrix Mult Benchmark"
	@echo "#####################################"
	./matrix_matrix_mult

pointwise_vector_max: pointwise_vector_max.cpp
	$(CXX) $(CXXFLAGS) pointwise_vector_max.cpp -o pointwise_vector_max $(AVXFLAGS)

pointwise_vector_max_build: pointwise_vector_max

pointwise_vector_max_run: pointwise_vector_max_build
	@echo "#####################################"
	@echo "# Pointwise Vector Max Benchmark"
	@echo "#####################################"
	./pointwise_vector_max

run: vector_norm_run vector_max_run pointwise_vector_max_run matrix_matrix_mult_run

clean:
	rm -f vector_norm_soa_plain
	rm -f vector_norm_soa_avx
	rm -f vector_norm_aos_plain
	rm -f vector_norm_aos_avx
	rm -f vector_max
	rm -f matrix_matrix_mult
	rm -f pointwise_vector_max
