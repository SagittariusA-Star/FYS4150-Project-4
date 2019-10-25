# include <mpi.h> 
# include <iostream>




int main(int argc, char *argv[])
{
    int numProcs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *sum = new int [2];
    int *result = new int [2];
    sum[0] = 1; sum[1] = 1;
    MPI_Allreduce(sum, result, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    for (int i = 0; i< 2; i++){
        std::cout << rank << " " << result[i] << " " << sum[i] << std::endl;

    }
    MPI_Finalize();

    return 0;
}