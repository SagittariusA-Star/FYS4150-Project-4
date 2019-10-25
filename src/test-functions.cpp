# include "catch.hpp"
# include "ising.cpp"


TEST_CASE("Comparing the expectation value for the energy and the heat capacity for the 2x2 lattice case with analytical solutions")
{   
    double tolerance = 1e-3;
    double E_mean;
    double E_var;
    int N = 2;
    int MC = 1e6;
    double T = 1.0;
    double *E = new double [MC];
    double *M = new double [MC];
    metropolis(MC, N, T, E, M);
    mean_and_variance(E, N, E_mean, E_var);
    REQUIRE(std::fabs((E_var / T * T) - Cv_2(T)) == Approx(0).epsilon(tolerance));
    REQUIRE(std::fabs(E_mean- E_mean_2(T)) == Approx(0).epsilon(tolerance));
}



