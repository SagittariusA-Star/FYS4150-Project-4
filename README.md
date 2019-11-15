# FYS4150-Project-4
Computational physics Project 4

Studies of phase transitions in magnetic systems

## Dependencies
* MPI
* Python >= 3.6
* LaTeX
&nbsp;

## Build Instructions:
* Run main_script.sh, this will build the source code, run tests and the main executables, generate plots and compile the report.
&nbsp;

## Structure
* main_script.sh, Main script that compiles, runs all codes and plots all results.
* src/ising.cpp, Main source code containing metropolis algorithm and various utility functions.
* src/task_a.cpp, Calculates values related to the test case with a 2 by 2 lattice
* src/task_b.cpp, Calculates themodynamical quantities for a 2 by 2 grid with T=1
* src/task_c_and_d.cpp, Sets temperature to 2.4, L=20, does same analysis as src/task_b.cpp. Then calculates the probability P(E)
* src/task_e.cpp, Runs a parallellized for L=40, 60, 80, 100. Takes a lot of time to run.
* src/timing_e.cpp, Times src/task_e.cpp for for different thread numbers.
* src/task_f.cpp, Estimates the critical temperature.
* src/plot_probability.py, Plots results from src/task_c_and_d.cpp
* src/plot_thermo_quants.py, Plots results from src/task_e.cpp
* src/plot_preprocess_big_data.py, Preprocesses results src/from task_c_and_d.cpp, for use in src/plot_probability.py
&nbsp;
