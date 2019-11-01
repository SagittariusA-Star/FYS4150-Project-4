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
    // Ex. 4a)
    int dummy_rank = 0;
    int N = 2;
    int MC = 1e5;
    double T = 1.0;
    double E_mean;
    double E_var;
    /*
    ofstream outfile;
    outfile.open("E_and_M_2by2.txt");
    outfile << " MC: " << " E: " << "M: " << endl;
    for (int i = 0; i < 7; i++)
    {   
        MC = (int) std::pow(10, i);
        cout << MC << endl;
        double *E = new double[MC];
        double *M = new double[MC];
        metropolis( MC, N, T, E, M, dummy_rank);
        mean_and_variance(E, MC, E_mean, E_var);
        outfile << setprecision(10) << setw(20) << MC
                << setprecision(10) << setw(20) << E_mean << endl;
        delete[] E;
        delete[] M;
    }
    outfile.close();

    */

   // Ex. 4b)
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

    //MPI_Finalize();
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
    
    // Ex. 4c)
    // Disordered lattice T = 1.0
    /*
    MC = 7;
    N = 20;
    int MC_len = 100;
    T = 1.0;
    double dMC = MC /((double) MC_len);
    double *MC_array = new double[MC_len];
    for (int i = 0; i < MC_len; i++)
    {
        MC_array[i] = dMC * i;
    }

    E_result = new double[MC_len];
    M_result = new double[MC_len];
    double *accp_result = new double[MC_len];

    matrix = lattice(N);
    
    results = new double[5];
    E_mean_array = new double[MC_len];
    M_mean_array = new double[MC_len];
    cout << "hei " << rank << endl;
    double *accp_array = new double[MC_len];

    for (int i = 0; i < MC_len; i++)
    {
        E_mean_array[i] = 0.0;
        M_mean_array[i] = 0.0;
        accp_array[i] = 0.0;
    }

    local_min = (int) std::round(MC_len * rank / (double) numbProc);
    local_max = (int) std::round(MC_len * (rank + 1) / (double) numbProc);
    int mc;
    for (int i = local_min; i < local_max; i++)    
    {   
        mc = std::pow(10, i);
        double *E = new double[mc];
        double *M = new double[mc];
        metropolis( mc, N, 0, matrix, T, E, M, results, rank);
        E_mean_array[i] = results[0];
        M_mean_array[i] = results[4];
        accp_array[i] = results[5];
    }
    MPI_Allreduce(E_mean_array, E_result, MC_len, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(M_mean_array, M_result, MC_len, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    delete[] E;
    delete[] M;
    delete[] results;
    delete[] E_mean_array;
    delete[] M_mean_array;
    delete[] accp_array;
    MPI_Finalize();

    ofstream outfile2;
    outfile2.open("disordered_T1.txt");
    outfile2 << " MC " << " <E>: " << "<|M|>: " << "#Accepted flips: " << endl;
    for (int i = 0; i < MC_len; i++)
    {   
        outfile2 << setprecision(10) << setw(20) << MC_array[i]
                << setprecision(10) << setw(20) << E_result[i]
                << setprecision(10) << setw(20) << M_result[i] 
                << setprecision(10) << setw(20) << accp_result[i] << endl;
    }
    outfile2.close();
    delete[] E_result;
    delete[] M_result;
    delete[] accp_result;
    
    */
    /*
    T = 1.0;
    rank = 0;
    MC = 1e6;
    N = 20;
    matrix = lattice(N);
    E = new double[MC];
    M = new double[MC];
    accp_flips = new double[MC];
    metropolis(MC, N, 0, matrix, T, E, M, accp_flips, results, rank);
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
    metropolis(MC, N, 0, matrix, T, E, M, accp_flips, results, rank);
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
    metropolis(MC, N, 0, matrix, T, E, M, accp_flips, results, rank);
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
    metropolis(MC, N, 0, matrix, T, E, M, accp_flips, results, rank);
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
*/  
    if (rank == 0)
    {
        cout << "Starting computation for L = 40" << endl;
    }

    MC = 1e6;
    N = 40;
    T_len = 50;
    T_min = 2.0;
    T_max = 2.5;
    dT = (T_max - T_min)/((double) T_len);
    T_array = new double[T_len];
    
    for (int i = 0; i < T_len; i++)
    {
        T_array[i] = T_min + dT * i;
    }

    C_result = new double[T_len];
    Chi_result = new double[T_len];
    E_result = new double[T_len];
    M_result = new double[T_len];

    matrix = lattice(N);

    results = new double[5];
    C_array = new double[T_len];
    Chi_array = new double[T_len];
    E_mean_array = new double[T_len];
    M_mean_array = new double[T_len];

    for (int i = 0; i < T_len; i++)
    {
        C_array[i] = 0.0;
        Chi_array[i] = 0.0;
        E_mean_array[i] = 0.0;
        M_mean_array[i] = 0.0;
    }

    E = new double[1];
    M = new double[1];
    accp_flips = new double[1]; 
    local_min = (int) std::round(T_len * rank / (double) numbProc);
    local_max = (int) std::round(T_len * (rank + 1) / (double) numbProc);

    for (int i = local_min; i < local_max; i++)    
    {   
        metropolis(MC, N, 5e3, matrix, T_array[i], E, M, accp_flips, results, rank);
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
        cout << "Printing to file: " << endl;
        ofstream outfile;
        outfile.open("L40.txt");
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

    if (rank == 0)
    {
        cout << "Starting computation for L = 60" << endl;
    }

    MC = 1e6;
    N = 60;
    T_len = 50;
    T_min = 2.0;
    T_max = 2.5;
    dT = (T_max - T_min)/((double) T_len);
    T_array = new double[T_len];
    
    for (int i = 0; i < T_len; i++)
    {
        T_array[i] = T_min + dT * i;
    }

    C_result = new double[T_len];
    Chi_result = new double[T_len];
    E_result = new double[T_len];
    M_result = new double[T_len];

    matrix = lattice(N);

    results = new double[5];
    C_array = new double[T_len];
    Chi_array = new double[T_len];
    E_mean_array = new double[T_len];
    M_mean_array = new double[T_len];

    for (int i = 0; i < T_len; i++)
    {
        C_array[i] = 0.0;
        Chi_array[i] = 0.0;
        E_mean_array[i] = 0.0;
        M_mean_array[i] = 0.0;
    }

    E = new double[1];
    M = new double[1];
    accp_flips = new double[1]; 
    local_min = (int) std::round(T_len * rank / (double) numbProc);
    local_max = (int) std::round(T_len * (rank + 1) / (double) numbProc);

    for (int i = local_min; i < local_max; i++)    
    {   
        metropolis(MC, N, 5e3, matrix, T_array[i], E, M, accp_flips, results, rank);
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
        cout << "Printing to file: " << endl;
        ofstream outfile;
        outfile.open("L60.txt");
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

    if (rank == 0)
    {
        cout << "Starting computation for L = 80" << endl;
    }

    MC = 1e6;
    N = 80;
    T_len = 50;
    T_min = 2.0;
    T_max = 2.5;
    dT = (T_max - T_min)/((double) T_len);
    T_array = new double[T_len];
    
    for (int i = 0; i < T_len; i++)
    {
        T_array[i] = T_min + dT * i;
    }

    C_result = new double[T_len];
    Chi_result = new double[T_len];
    E_result = new double[T_len];
    M_result = new double[T_len];

    matrix = lattice(N);

    results = new double[5];
    C_array = new double[T_len];
    Chi_array = new double[T_len];
    E_mean_array = new double[T_len];
    M_mean_array = new double[T_len];

    for (int i = 0; i < T_len; i++)
    {
        C_array[i] = 0.0;
        Chi_array[i] = 0.0;
        E_mean_array[i] = 0.0;
        M_mean_array[i] = 0.0;
    }

    E = new double[1];
    M = new double[1];
    accp_flips = new double[1]; 
    local_min = (int) std::round(T_len * rank / (double) numbProc);
    local_max = (int) std::round(T_len * (rank + 1) / (double) numbProc);

    for (int i = local_min; i < local_max; i++)    
    {   
        metropolis(MC, N, 5e3, matrix, T_array[i], E, M, accp_flips, results, rank);
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
        cout << "Printing to file: " << endl;
        ofstream outfile;
        outfile.open("L80.txt");
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

    if (rank == 0)
    {
        cout << "Starting computation for L = 100" << endl;
    }

    MC = 1e6;
    N = 100;
    T_len = 50;
    T_min = 2.0;
    T_max = 2.5;
    dT = (T_max - T_min)/((double) T_len);
    T_array = new double[T_len];
    
    for (int i = 0; i < T_len; i++)
    {
        T_array[i] = T_min + dT * i;
    }

    C_result = new double[T_len];
    Chi_result = new double[T_len];
    E_result = new double[T_len];
    M_result = new double[T_len];

    matrix = lattice(N);

    results = new double[5];
    C_array = new double[T_len];
    Chi_array = new double[T_len];
    E_mean_array = new double[T_len];
    M_mean_array = new double[T_len];

    for (int i = 0; i < T_len; i++)
    {
        C_array[i] = 0.0;
        Chi_array[i] = 0.0;
        E_mean_array[i] = 0.0;
        M_mean_array[i] = 0.0;
    }

    E = new double[1];
    M = new double[1];
    accp_flips = new double[1]; 
    local_min = (int) std::round(T_len * rank / (double) numbProc);
    local_max = (int) std::round(T_len * (rank + 1) / (double) numbProc);

    for (int i = local_min; i < local_max; i++)    
    {   
        metropolis(MC, N, 5e3, matrix, T_array[i], E, M, accp_flips, results, rank);
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
        cout << "Printing to file: " << endl;
        ofstream outfile;
        outfile.open("L100.txt");
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