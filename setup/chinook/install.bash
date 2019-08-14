#!/bin/bash -e

# Creates MTUQ virtual environment on chinook


if [[ ! $HOSTNAME == chinook* ]];
then
    echo ""
    echo "Error: This script works only on chinook*.alaska.edu"
    echo ""
    exit 1
fi


# navigate to mtuq/setup/chinook
cd $(dirname ${BASH_SOURCE[0]})


DIR="$PWD/install/mtuq"
if [[ -d $DIR ]];
then
    echo ""
    echo "Error: Virtual environment already exists"
    echo ""
    exit 1
fi


# load system modules
module load lang/Python/2.7.12-pic-intel-2016b


# create virutal environment
mkdir -p $DIR
virtualenv "$DIR"
source "${DIR}/bin/activate"
pip --no-cache-dir install mpi4py
pip --no-cache-dir install numpy
pip --no-cache-dir install obspy instaseis


# install mtuq in editable mode
cd ../..
pip --no-cache-dir install -e .


# adjust matplotlib backend
find . -name matplotlibrc -exec sed -i '/backend *:/s/TkAgg/Agg/' {} +


# unpack examples
./data/examples/unpack.bash
./data/tests/unpack.bash


deactivate


