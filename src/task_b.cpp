# include <fstream>
# include <iostream>
# include "ising.cpp"
# include <stdexcept>
# include <iomanip>
# include <armadillo>
# include <mpi.h>
using std::cout;
using std::ofstream;
using std::setprecision;
using std::setw;

int main(int argc, char *argv[])
{
  int N;
  int MC;

  MC = 1e6;
  N = 2;
  int T_len = 5;
  double T_min = 1.0;
  double T_max = 2.4;
  double dT = (T_max - T_min)/((double) T_len);

  int numbProc;
  int rank;

  double *T_array = new double[T_len];

  for (int i = 0; i < T_len; i++)
  {
      T_array[i] = T_min + dT * i;
  }

  double *C_result = new double[T_len];
  double *Chi_result = new double[T_len];
  double *E_result = new double[T_len];
  double *M_result = new double[T_len];

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numbProc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  arma::imat matrix = lattice(N);
  double *results = new double[5];
  double *C_array = new double[T_len];
  double *Chi_array = new double[T_len];
  double *E_mean_array = new double[T_len];
  double *M_mean_array = new double[T_len];

  for (int i = 0; i < T_len; i++)
  {
      C_array[i] = 0.0;
      Chi_array[i] = 0.0;
      E_mean_array[i] = 0.0;
      M_mean_array[i] = 0.0;
  }

  double *E = new double[1];
  double *M = new double[1];
  double *accp_flips = new double[1];
  double local_min = (int) std::round(T_len * rank / (double) numbProc);
  double local_max = (int) std::round(T_len * (rank + 1) / (double) numbProc);

  for (int i = local_min; i < local_max; i++)
  {
      metropolis(MC, N, 0, matrix, T_array[i], E, M, accp_flips, results, rank);
      E_mean_array[i] = results[0];
      M_mean_array[i] = results[4];
      C_array[i] = results[1] / (T_array[i] * T_array[i]);
      Chi_array[i] = results[3] / T_array[i];

  }
  MPI_Allreduce(C_array, C_result, T_len, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(Chi_array, Chi_result, T_len, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(E_mean_array, E_result, T_len, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(M_mean_array, M_result, T_len, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  delete[] E;
  delete[] M;
  delete[] accp_flips;
  delete[] results;
  delete[] C_array;
  delete[] Chi_array;
  delete[] E_mean_array;
  delete[] M_mean_array;

  if (rank == 0)
  {
      ofstream outfile;
      outfile.open("Ex_4b.txt");
      outfile << " T: " << " <E>: " << "<|M|>: " << "C_V: " << "Chi: " << endl;
      for (int i = 0; i < T_len; i++)
      {
          outfile << setprecision(10) << setw(20) << T_array[i]
                  << setprecision(10) << setw(20) << E_result[i]
                  << setprecision(10) << setw(20) << M_result[i]
                  << setprecision(10) << setw(20) << C_result[i]
                  << setprecision(10) << setw(20) << Chi_result[i] << endl;
      }
      outfile.close();
      delete[] C_result;
      delete[] Chi_result;
      delete[] E_result;
      delete[] M_result;
  }
   MPI_Finalize();
}
