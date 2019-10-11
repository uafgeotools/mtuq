#!/bin/bash

# Activates MTUQ virtual environment on chinook
#
# Invoke this script by sourcing it, i.e.
# > source activate.bash


if [ "${BASH_SOURCE[0]}" == "${0}" ];
then
    echo ""
    echo "This script must be sourced"
    echo ""
    exit 1
fi


if [[ ! $HOSTNAME == chinook* ]];
then
    echo ""
    echo "This script works only on chinook*.alaska.edu"
    echo ""
    return 1
fi


# what is the relative path to mtuq/setup/chinook?
SETUP=$(dirname ${BASH_SOURCE[0]})


ERROR="
Virtual environment not found.
Run mtuq/setup/chinook/install.bash and try again.
"
if [[ ! -d $SETUP/install/mtuq  ]];
then
    echo "$ERROR"
    return 1
fi


# load system modules
module load lang/Python/2.7.12-pic-intel-2016b
export CC=icc


# activate virutal environment
source $SETUP/install/$VENV/mtuq/bin/activate


