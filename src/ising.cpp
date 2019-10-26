# include "ising.h"
# include <stdexcept>
using std::cout;
using std::exp;
using std::endl;


inline int periodic_index(int i, int N)
{
    return (i + N) % N;
}

double E_init(arma::imat lattice)
{
    int E = 0;
    double peri_index_i;
    double peri_index_j;
    int N_row = lattice.n_rows;
    int N_col = lattice.n_cols;
    for (int i = 0; i < N_row; i++) {
        peri_index_i = periodic_index(i - 1, N_row);
    for (int j = 0; j < N_col; j++) {
        peri_index_j = periodic_index(j - 1, N_col);
        E += lattice(i, j) * (lattice(peri_index_i, j) +
             lattice(i, peri_index_j));
    }}
    E *= - 1;
    return E;
}

double M_init(arma::imat lattice)
{
    return arma::sum(arma::sum(lattice, 0));
}

arma::imat lattice (int N)
{   
    arma::arma_rng::set_seed(clock());
    arma::imat lattice = arma::randi<arma::imat>(N, N, arma::distr_param(0,1));
    for (int i=0; i < N; i++) {
    for (int j=0; j < N; j++) {
        if (lattice(i,j) == 0) 
        {
            lattice(i,j) = -1;
        }
    }}
    return lattice;
}

void mean_and_variance(double *A, int N, double &mean, double &var)
/*
    Function comuting mean of given array.
    
    Parameters:
    -----------
    A: double *
        Array to compute mean  and variance of.
    N: int 
        Length of array.
    mean: double
        Mean that is to be filled.
    var: double
        Variance to be filled.
*/
{   
    mean = 0;
    var = 0;
    for (int i = 0; i < N; i++)
    {
        mean += A[i];
        var += A[i] * A[i];
    }
    mean /= (double) N;
    var /= (double) N;
    var -= mean;
}

double Cv_2(double T)
/*
Calculate the anaylitical expression for the
heat capacity for a 2x2 lattice
Parameters
------------
T: double
    Temperature in units k_B * T / J
*/
{
    double Cv = 192 * (std::cosh(8.0 / T) + 1)
                    / (T * T * std::pow(std::cosh(8.0 / T) + 3, 2));
    return Cv;      
}

double  E_mean_2(double T)
/*
Calculates the analytical expectation value for the energy
for a 2x2 lattice.

Parameters
------------
T: double
    Temperature in units k_B * T / J
*/
{
    double expval = -8 * std::sinh(8.0 / T) / (std::cosh(8.0 / T) + 3.0);
    return expval;
}

void metropolis(int MC, int N, int start_samp, arma::imat &matrix,
                double T, double *E, double *M, double *results, int rank)
/*
----------
MC: int
    Number of Monte Carlo cycles.
N: int 
    Dimension of lattice.
start_samp: int
    Number of Monte Carlo cycles after which to sample.
matrix: arma::imat
    Lattice of spins.
T: double
    Temperature in units k_B * T / J.
E: double
    Vector of length MC with energies.
M: double
    Vector of lengths MC with magnetic moments.
results: double
    Vector of length 6 to fill with E/MC, E*E/MC, M/MC, M*M/MC,
    fabs(M)/MC and accepted flip numbers.
rank: int
    Rank of processor.
*/
{   
    std::mt19937_64 generator;
    std::uniform_int_distribution<int> distribution(0, N - 1);
    std::uniform_real_distribution<double> accepting(0, 1);
    generator.seed(MPI_Wtime() + rank);
    int i_samp;
    int j_samp;
    double delta_E;
    arma::vec boltzmann_precal = arma::zeros<arma::vec>(17);
    boltzmann_precal(0)  = exp(8.0 / T);
    boltzmann_precal(4)  = exp(4.0 / T);
    boltzmann_precal(8)  = 1.0;
    boltzmann_precal(12) = exp(-4.0 / T);
    boltzmann_precal(16) = exp(-8.0 / T);

    E[0] = E_init(matrix);
    M[0] = M_init(matrix);
    double _E = E[0]; 
    double _M = M[0];
    double accepted_flip = 0.0;
    results[0] = 0.0;
    results[1] = 0.0;
    results[2] = 0.0;
    results[3] = 0.0;
    results[4] = 0.0;
    results[5] = 0.0;

    for (int i = 0; i < MC; i++){
        for (int j = 0; j < N * N; j++){  
            i_samp = distribution(generator);
            j_samp = distribution(generator);
            delta_E = 2 * matrix(i_samp, j_samp)
                        *(matrix(periodic_index(i_samp + 1, N), j_samp)
                        + matrix(periodic_index(i_samp - 1, N), j_samp)
                        + matrix(i_samp, periodic_index(j_samp + 1, N))
                        + matrix(i_samp, periodic_index(j_samp - 1, N)));
            if (boltzmann_precal((int) delta_E + 8) >= accepting(generator))
            {   
                matrix(i_samp, j_samp) *= - 1;
                _E += delta_E;
                _M += 2 * matrix(i_samp, j_samp);
                accepted_flip += 1;
                
            }
        }
        if (i >= start_samp)
            {
                results[0] += _E;
                results[1] += _E * _E;
                results[2] += _M;
                results[3] += _M * _M;
                results[4] += std::fabs(_M);
                results[5] += accepted_flip;
            }
        E[i] = _E;
        M[i] = _M;        
    }
    results[0] /= (double) MC;
    results[1] /= (double) MC;
    results[1] -= results[0] * results[0];
    results[2] /= (double) MC;
    results[3] /= (double) MC;
    results[3] -= results[2] * results[2];
    results[4] /= (double) MC;
}

