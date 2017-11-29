MPICXX= mpic++
MPICXXFLAGS= -O2 -std=c++11

all: matrix_mult_2D matrix_mult_cols matrix_mult_rows summa

matrix_mult_2D: matrix_mult_2D.cpp
	$(MPICXX) $(MPICXXFLAGS) matrix_mult_2D.cpp -o matrix_mult_2D

matrix_mult_cols: matrix_mult_cols.cpp
	$(MPICXX) $(MPICXXFLAGS) matrix_mult_cols.cpp -o matrix_mult_cols

matrix_mult_rows: matrix_mult_rows.cpp
	$(MPICXX) $(MPICXXFLAGS) matrix_mult_rows.cpp -o matrix_mult_rows

summa: summa.cpp
	$(MPICXX) $(MPICXXFLAGS) summa.cpp -o summa

clean:
	rm -rf matrix_mult_2D
	rm -rf matrix_mult_cols
	rm -rf matrix_mult_rows
	rm -rf summa
