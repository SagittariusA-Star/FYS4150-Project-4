# include "catch.hpp"
# include "ising.cpp"


TEST_CASE()
{   
    double tolerance = 1e-3;
    double E_mean;
    int N = 2;
    int MC = 1e6;
    double T = 1.0;
    double *E = new double [MC];
    double *M = new double [MC];
    metropolis(MC, N, T, E, M);

    for (int i=0; i<MC; i++)
    {
        E_mean += E[i];
    }
    E_mean /= (double)MC;
    cout << E_mean << " " << E_mean_2(T) << endl;
    REQUIRE(std::fabs(E_mean- E_mean_2(T)) == Approx(0).epsilon(tolerance));
}



