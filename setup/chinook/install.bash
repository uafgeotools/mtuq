#!/bin/bash -e

# Creates MTUQ virtual environment on chinook


if [[ ! $HOSTNAME == chinook* ]];
then
    echo "Error: This script works only on chinook.alaska.edu"
    exit 1
fi


# navigate to mtuq/setup/chinook
cd $(dirname ${BASH_SOURCE[0]})
VDIR="$PWD/install"
VENV="mtuq"

if [[ -d ${VIDR}/${VENV} ]];
then
    echo "Error: Virtual environment already exists"
    exit 1
fi


# load system modules
module load lang/Python/2.7.12-pic-intel-2016b


# create virutal environment
mkdir -p $VDIR
virtualenv "${VDIR}/${VENV}"
source "${VDIR}/${VENV}/bin/activate"


# install mtuq in editable mode
cd "../.."
pip install numpy
pip install -e .
pip install mpi4py


# adjust matplotlib backend
find . -name matplotlibrc -exec sed -i '/backend *:/s/TkAgg/Agg/' {} +


# unpack examples
./data/examples/unpack.bash
./data/tests/unpack.bash


deactivate


