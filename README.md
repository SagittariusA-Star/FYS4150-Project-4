# FYS4150-Project-4
Computational physics Project 4

## Dependencies
* MPI
* Python >= 3.6
* LaTeX
&nbsp;

## Build Instructions:
* Run main_script.sh, this will build the source code, run tests and the main executables, generate plots and compile the report.
&nbsp;

## Structure
* main_script.sh
* src/ising.cpp, Main source code containing metropolis algorithm and various utility functions.
* src/task_a.cpp, Calculates values related to the test case with a 2 by 2 lattice
* src/task_b.cpp, Calculates themodynamical quantities for a 2 by 2 grid with T=1
* src/task_c_and_d.cpp, Sets temperature to 2.4, L=20, does same analysis as src/task_b.cpp. Then calculates the probability P(E)
* src/task_e.cpp, Runs a parallellized for L=40, 60, 80, 100. Takes a lot of time to run.
* src/task_f.cpp, Etimates the critical temperature.
&nbsp;
