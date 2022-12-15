## The problem statement can be found [here](https://www.cse.iitd.ac.in/~rijurekha/col730_2022/mpi_assignment.html).

### The project consists of three parts:
#### Part 1: cdmh MapReduce Library
Implementing PageRank problem in the given [library](https://github.com/cdmh/mapreduce) with the help of the provided templates for a MapTask and a ReduceTask.The null combiner given in the library was used as this is sequential.
 
#### Part 2: My Implementation of PageRank Algorithm (using MPI)
I implemented this using MPI. I used the MPICH implementation of the MPI standard. My mapper function stored the value of each new element of Ip
in the vector itself, rather than creating key-value pairs. We can see this as the index being the key. Then for the reduce step, I called the Allreduce operation on the whole vector, as that would handle communication between the processors optimally.

#### Part 3: Map-Reduce MPI Library
Using the MPI Map Reduce library, I had to write my map task and my reduce task in the form of 2 functions, that are arguments to the library’s map and reduce function and are called internally on each key, so that the library can handle the parallelism (where to send each key). This was the baseline against which my implementation was evaluated. 

The implementations were tested on the PageRank benchmarks provided [here](https://github.com/louridas/pagerank/tree/master/test). Performance was assessed by comparing the PageRank latencies across the different implementation (the results can be viewed in the report).
