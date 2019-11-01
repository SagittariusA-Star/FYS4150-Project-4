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

int main()
{
    // Ex. 4a)
    int dummy_rank = 0;
    int N = 2;
    int MC;
    double T = 1.0;
    arma::imat lattice_matrix = lattice(N);
    double *results = new double[5];

    ofstream outfile;
    outfile.open("E_and_M_2by2.txt");
    outfile << " MC: " << " E: " << "M: " << endl;
    for (int i = 0; i < 7; i++)
    {
        MC = (int) std::pow(10, i);
        cout << MC << endl;
        double *E = new double[MC];
        double *M = new double[MC];
        double *accp_flip = new double[MC];
        metropolis(
          MC,
          N,
          0,
          lattice_matrix,
          T,
          E,
          M,
          accp_flip,
          results,
          dummy_rank,
          true
        );
        outfile << setprecision(10) << setw(20) << MC
                << setprecision(10) << setw(20) << results[0] << endl;
        delete[] E;
        delete[] M;
        delete[] accp_flip;
    }
    outfile.close();
    delete[] results;

}
