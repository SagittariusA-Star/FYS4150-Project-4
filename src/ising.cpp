# include <iostream>
# include <fstream>
//# include <mpi.h> 
# include <armadillo>
# include <ctime>

using std::cout;
using std::endl;



double E_init(arma::imat lattice, double J)
{
    int E = 0;
    double peri_index_i;
    double peri_index_j;
    int N_row = lattice.n_rows;
    int N_col = lattice.n_cols;
    for (int i=0; i<N_row; i++) {
        peri_index_i = (i + N_row) % N_row;
    for (int j=0; j<N_col; j++) {
        peri_index_j = (j + N_col) % N_col;
        E += lattice(i, j) * (lattice(peri_index_i, j) +
             lattice(i, peri_index_j));
    }}
    E *= - J;
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
    for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
        if (lattice(i,j) == 0) 
        {
            lattice(i,j) = -1;
        }
    }}
    lattice.print();
    return lattice;
}


int main ()
{   
    int N = 2;
    arma::imat testlattice = lattice(N);
    double E = E_init(testlattice, 1);
    double M = M_init(testlattice);
    cout << E << " " << M << endl;
    return 0;
}