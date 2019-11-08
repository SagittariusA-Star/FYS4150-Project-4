#!/bin/bash
cd src

make clean
make rmexecs
make

echo "Run tests? (y/n"
read yntest

if [ "$yntest" == "y" ]
then
    ./testcode.out
fi

echo "Generate new data for task a?  (y/n)"
read yn
if [ "$yn" == "y" ]
then
    ./task_a.out
fi

echo "Generate new data for task b?  (y/n)"
read yn
if [ "$yn" == "y" ]
then
    echo "How many threads would you like to run the program with?"
    read nthread
    ./task_b.out -n $nthread
fi

echo "Generate new data for task c? This may take several hours (y/n)"
read yn
if [ "$yn" == "y" ]
then
    echo "How many threads would you like to run the program with?"
    read nthread
    ./task_c_d.out -n $nthread
    python3 preprocess_big_data.py
fi



echo "Generate plots? (y/n)"
read yn
if [ $"yn" == "y" ]
then 
    python3 plot_probability.py
    python3 plot_thermo_quants.py
    python3 plot_2by_lattice.py
fi

