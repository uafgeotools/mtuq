#!/bin/bash

cd $(dirname ${BASH_SOURCE[0]})

rm -rf build/*
rm -f mtuq/misfit/*.so

python ./setup.py build_ext --inplace

