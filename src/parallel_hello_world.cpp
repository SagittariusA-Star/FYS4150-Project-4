# include <mpi.h>
# include <iostream>

# include "ising.cpp"



int main(int argc, char *argv[])
{
    int numProcs, rank;
    int N = 2;
    int MC = 10;
    //double *summed_E = new double [MC];
    double T = 1.0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double E_mean;
    double E_var;
    double *E = new double [MC];
    double *M = new double [MC];
    double *summed_E = new double [MC];
    metropolis(MC, N, T, E, M);
    mean_and_variance(E, N, E_mean, E_var);


    MPI_Allreduce(E, summed_E, MC, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i< MC; i++){
        std::cout << "i: " << i << " Thread: " << rank << " Local E " << E[i] << " Summed E: " << summed_E[i] << std::endl;

    }
    MPI_Finalize();
    return 0;
}
