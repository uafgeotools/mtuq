#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --time=15


# error checking
ERROR="
Virtual environment not activated

source activate.bash and try again
"
if [[ $( pip -V ) != *mtuq* ]];
then
    echo "$ERROR"
    exit
fi


ERROR="
Wrong number of input arguments

USAGE
    sbatch submit.bash name_of_example
"
if [ $# -ne 1 ];
then
    echo "$ERROR"
    exit
fi


# run example
mpirun -n 24 python $1


