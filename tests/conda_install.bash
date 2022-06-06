#!/bin/bash


#
# TESTS MTUQ INSTALLATION UNDER CONDA
#


#
# mtuq root directory
#
MTUQ_PATH=$(dirname ${BASH_SOURCE[0]})/..


#
# check that the following versions and dependencies match 
# mtuq/docs/install/env_conda.rst
#
PYTHON_VERSION=3
DEPENDENCIES="scipy obspy instaseis pandas xarray netCDF4 h5py mpi4py"


#
# path to existing conda installation, or if not already present, where
# conda will be installed using the functions below
#
CONDA_PATH="$HOME/miniconda3"


function conda_install {
    CONDA_PATH=$1

    hash -r
    wget -nv $(conda_url) -O miniconda.sh
    bash miniconda.sh -b -f -p $CONDA_PATH
    hash -r
    rm miniconda.sh
}


function conda_update {
    CONDA_PATH=$1

    conda config --set always_yes yes --set changeps1 no 
    conda update -q conda
    conda info -a
    conda config --add channels conda-forge
}


function conda_url {
case "$(uname -s)" in
   Darwin)
     URL="https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
     ;;
   Linux)
     URL="https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
     ;;
   CYGWIN*|MINGW32*|MSYS*|MINGW*)
     URL="https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.sh"
     ;;
   *)
     echo "OS not recognized"
     exit -1
     ;;
esac
echo $URL
}


#
# installation tests begin now
#

# if any test fails, stop immediately
set -e

echo
echo "See mtuq/tests/ for installation logs"
echo
cd $MTUQ_PATH

echo
echo "Installing latest version of conda"
echo
[ -d $CONDA_PATH ] || conda_install $CONDA_PATH > tests/log1
source $CONDA_PATH/etc/profile.d/conda.sh
conda_update $CONDA_PATH >> tests/log1
echo SUCCESS
echo

echo "Testing mtuq installation without PyGMT"
conda create -q -n env_step2 python=$PYTHON_VERSION > tests/log2
conda activate env_step2
conda install $DEPENDENCIES >> tests/log2
pip install -e . >> tests/log2
conda deactivate
echo SUCCESS
echo 

echo "Testing mtuq installation with PyGMT"
conda create -q -n env_step3 python=$PYTHON_VERSION > tests/log3
conda activate env_step3
conda install ${DEPENDENCIES} >> tests/log3
pip install -e . >> tests/log3
conda install pygmt >> tests/log3
conda deactivate
echo SUCCESS
echo

