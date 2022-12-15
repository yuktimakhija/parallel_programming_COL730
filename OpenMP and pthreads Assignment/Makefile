define arguments_help 

Yukti Makhija, 2019BB10067
COL730 Assignment 2.

This makefile will compile the OpenMP and pthread code into the binary res_omp_bb1190067 and res_pth_bb1190067.

It needs the parameter:
	threads - the number of threads 
	n - size of the matrix

Example usage: ./res_omp_bb1190067 16 4000
endef

export arguments_help
all:
	@echo "$$arguments_help"
	@gcc matrix_lud_pointer_openmp.c -o res_omp_bb1190067 -lm -fopenmp -std=c99
	@gcc matrix_lud_pointer_pthread.c -o res_pth_bb1190067 -lm -fopenmp -pthread -std=c99
	@echo Compilation Ended, Exiting...
