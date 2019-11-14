#!/bin/bash
cd src

echo "Compile? (y/n)"
read yn
if [ "$yn" == "y" ]
then
  make clean
  make rmexecs
  make
fi

echo "Run tests? (y/n)"
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
    mpirun ./task_b.out -n $nthread
fi

echo "Generate new data for task c and d? This may take several hours (y/n)"
read yn
if [ "$yn" == "y" ]
then
    echo "How many threads would you like to run the program with?"
    read nthread
    mpirun ./task_c_d.out -n $nthread
    python3 preprocess_big_data.py
fi

echo "Generate new data for task e? (y/n)"
read yn
if [ "$yn" == "y" ]
then
    echo "How many threads would you like to run the program with?"
    read nthread
    mpirun ./task_e.out -n $nthread
    python3 preprocess_big_data.py
fi

echo "Generate plots? (y/n)"
read yn
if [ "$yn" == "y" ]
then
    python3 plot_probability.py
    python3 plot_thermo_quants.py
fi

echo "Compile report? (y/n)"
read yn
if [ "$yn" == "y" ]
then
  cd ../doc/
  pdflatex -synctex=1 -interaction=nonstopmode CompPhysProj4.tex
  bibtex CompPhysProj4.aux
  pdflatex -synctex=1 -interaction=nonstopmode CompPhysProj4.tex
  bibtex CompPhysProj4.aux
  pdflatex -synctex=1 -interaction=nonstopmode CompPhysProj4.tex
fi
