# include <mpi.h> 
# include <iostream>




int main(int argc, char *argv[])
{
    int numProcs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int sum = 2;

    MPI_Reduce(&rank, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    std::cout << rank << " " << sum << std::endl;
    MPI_Finalize();

    return 0;
}