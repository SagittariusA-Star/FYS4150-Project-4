# include "catch.hpp"
# include "ising.hpp"

TEST_CASE("Compares the expectation value for the energy, heat capacity, mean magnetization and for the 2x2 lattice case with analytical solutions")
{   
    double tolerance = 1e-3;
    int start_samp = 0;
    int N = 2;
    int MC = 5e7;
    double T = 1.0;
    double *E = new double [1];
    double *M = new double [1];
    double *accp_flip = new double [1];
    double *results = new double[6];
    int rank = 0; 
    arma::imat matrix = lattice(N);
    metropolis(MC, N, start_samp, matrix, T, E, M, accp_flip, results, rank);
    REQUIRE(std::fabs(results[0] - E_mean_2(T)) == Approx(0).epsilon(tolerance));
    REQUIRE(std::fabs((results[1] / (T * T)) - Cv_2(T)) == Approx(0).epsilon(tolerance));
    REQUIRE(std::fabs(results[3] / (T) - susc_2(T)) == Approx(0).epsilon(tolerance));
    REQUIRE(std::fabs(results[4] - M_mean_2(T)) == Approx(0).epsilon(tolerance));
    delete[] E;
    delete[] M;
    delete[] results;
}

TEST_CASE("Compares the numerical initial energy and magnetisation with known values for 2x2 lattice")
{   
    int N = 2;
    double E;
    double M; 
    arma::imat matrix = arma::ones<arma::imat>(N, N);
    E = E_init(matrix);
    M = M_init(matrix);
    REQUIRE(E == -8);
    REQUIRE(M == 4);

    matrix(0,0) = -1;
    E = E_init(matrix);
    M = M_init(matrix);
    REQUIRE(E == 0);
    REQUIRE(M == 2);

    matrix = arma::ones<arma::imat>(N, N);
    matrix(1,0) = -1;
    E = E_init(matrix);
    M = M_init(matrix);
    REQUIRE(E == 0);
    REQUIRE(M == 2);
}

