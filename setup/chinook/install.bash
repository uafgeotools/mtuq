#!/bin/bash -e

# Creates MTUQ virtual environment on chinook


if [[ $HOSTNAME != chinook* ]];
then
    echo "Error: This script works only on chinook.alaska.edu"
    exit 1
fi


# load system modules
module load lang/Python/2.7.12-pic-intel-2016b


# navigate to mtuq/setup/chinook
cd $(dirname ${BASH_SOURCE[0]})
VDIR="$PWD/virtual"
VENV="mtuq"


if [[ -d $SETUP/virtual/mtuq  ]];
then
    echo "Error: Virtual environment already exists"
    exit 1
fi


# create virutal environment
cd $VDIR
virtualenv $VENV
source "$VENV/bin/activate"


# install dependencies
pip install numpy
pip install mpi4py


# install mtuq in editable mode
cd "../.."
pip install -e .


# unpack examples
./data/examples/unpack.bash
./data/tests/unpack.bash


deactivate


