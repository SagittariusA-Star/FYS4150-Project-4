# ifndef MAIN_H
# define MAIN_H

# include "ising.cpp"


inline int periodic_index(int, int);

double E_init(arma::imat);

double M_init(arma::imat);

arma::imat lattice (int N);

                
double Cv_2(double);

double  E_mean_2(double);

double M_mean_2(double);

double susc_2(double T);

void metropolis(int, int, int, arma::imat &,
                double, double *, double *, double *, 
                double *, int, bool);

#endif 


