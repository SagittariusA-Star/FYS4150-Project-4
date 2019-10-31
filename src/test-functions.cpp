# include "catch.hpp"
# include "ising.cpp"
# include <armadillo>

TEST_CASE("Comparing the expectation value for the energy and the heat capacity for the 2x2 lattice case with analytical solutions")
{   
    double tolerance = 1e-3;
    double E_mean;
    double E_var;
    int N = 2;
    int MC = 1e7;
    double T = 1.0;
    double *E = new double [MC];
    double *M = new double [MC];
    double *results = new double[6];
    int rank = 0;
    arma::imat matrix = lattice(N);
    metropolis(MC, N, 0, matrix, T, E, M, results, rank);
    cout << results[1] << " " << Cv_2(T) << endl;
    cout << results[0] << " " << E_mean_2(T) << endl;
    REQUIRE(std::fabs((results[1] / (T * T)) - Cv_2(T)) == Approx(0).epsilon(tolerance));
    REQUIRE(std::fabs(results[0] - E_mean_2(T)) == Approx(0).epsilon(tolerance));
    delete[] E;
    delete[] M;
    delete[] results;
}



