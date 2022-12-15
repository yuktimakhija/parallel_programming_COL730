all:
	g++ mr-pr-cpp.cpp -o mr-pr-cpp.o -std=c++11 -lboost_filesystem -lboost_iostreams -lpthread
	mpicxx mr-pr-mpi.cpp -o mr-pr-mpi.o
	mpicxx mr-pr-mpi-base.cpp -o mr-pr-mpi-base.o -I MPI-mapreduce/ -O MPI-mapreduce/libmrmpi_mpicc.a