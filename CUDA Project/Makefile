define arguments_help 

Yukti Makhija, 2019BB10067
COL730 Assignment 1.

This makefile will compile the CUDA code into the binary kernels_fann_bb1190067.
It is expected that the fann library is installed as per general install instructions given on the website.
Run this Makefile and the kernels_fann_bb1190067 file from the examples folder inside fann.
It needs the parameter:
	dataset - Should be one of the strings mushroom | robot | kin32fm
			corresponding to whatever dataset needs to be trained on.

Example usage: ./kernels_fann_bb1190067 mushroom
endef

export arguments_help
all:
	@echo "$$arguments_help"
	@nvcc kernels_optimized.cu -lfann -o kernels_fann_bb1190067 2>/dev/null
	@echo Compilation Ended, Exiting...
