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
    int N = 20;
    int MC = 1e7; 
    double T = 1.0;
    int rank = 0;

    arma::imat matrix = lattice(N);
    double *E = new double[MC];
    double *M = new double[MC];
    double *accp_flips = new double[MC];
    double *results = new double[5];
    metropolis(MC, N, 0, matrix, T, E, M, accp_flips, results, rank, true);
    ofstream outfile3;
    outfile3.open("P_T1_MC1e7_Disordered.txt");
    outfile3 << "E: " << "|M|" << "Accepted flips" <<  "E_var (Next line): " << endl;
    outfile3 << results[1] << endl;
    for (int i = 0; i < MC; i++)
    {   
        outfile3 << setprecision(10) << setw(20) << E[i]
                 << setprecision(10) << setw(20) << std::fabs(M[i])
                 << setprecision(10) << setw(20) << accp_flips[i]
                 << endl;
    }
    outfile3.close();

    matrix.fill(1);
    metropolis(MC, N, 0, matrix, T, E, M, accp_flips, results, rank, true);
    ofstream outfile4;
    outfile4.open("P_T1_MC1e7_Ordered.txt");
    outfile4 << "E: " << "|M|" << "Accepted flips" <<  "E_var (Next line): " << endl;
    outfile4 << results[1] << endl;
    for (int i = 0; i < MC; i++)
    {
        outfile4 << setprecision(10) << setw(20) << E[i]
                 << setprecision(10) << setw(20) << std::fabs(M[i])
                 << setprecision(10) << setw(20) << accp_flips[i]
                 << endl;
    }
    outfile4.close();


    T = 2.4;
    matrix = lattice(N);
    metropolis(MC, N, 0, matrix, T, E, M, accp_flips, results, rank, true);
    ofstream outfile5;
    outfile5.open("P_T24_MC1e7_Disordered.txt");
    outfile5 << "E: " << "|M|" << "Accepted flips" <<  "E_var (Next line): " << endl;
    outfile5 << results[1] << endl;
    for (int i = 0; i < MC; i++)
    {
        outfile5 << setprecision(10) << setw(20) << E[i]
                 << setprecision(10) << setw(20) << std::fabs(M[i])
                 << setprecision(10) << setw(20) << accp_flips[i]
                 << endl;
    }
    outfile5.close();

    matrix.fill(1);
    metropolis(MC, N, 0, matrix, T, E, M, accp_flips, results, rank, true);
    ofstream outfile6;
    outfile6.open("P_T24_MC1e7_Ordered.txt");
    outfile6 << "E: " << "|M|" << "Accepted flips" <<  "E_var (Next line): " << endl;
    outfile6 << results[1] << endl;
    for (int i = 0; i < MC; i++)
    {
        outfile6 << setprecision(10) << setw(20) << E[i]
                 << setprecision(10) << setw(20) << std::fabs(M[i])
                 << setprecision(10) << setw(20) << accp_flips[i]
                 << endl;
    }
    outfile6.close();
    delete[] E;
    delete[] M;
    delete[] accp_flips;

}
