#!/bin/bash
#SBATCH --parition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --time=15


# error checking
ERROR="
This script works only on chinook.alaska.edu
"
if [[ $HOSTNAME != chinook* ]];
then
    echo "$ERROR"
    exit
fi


ERROR="
Virtual environment not activated.
Run mtuq/setup/chinook/activate.bash and try again
"
if [[ pip -V != *mtuq* ]];
then
    echo "$ERROR"
    exit
fi


ERROR="
USAGE
    sbatch submit.bash CapStyleGridSearch.DoubleCouple.py
"
if [ $? -ne 1 ];
then
    echo "$ERROR"
    exit
fi


# run example
mpirun -np 24 $1


