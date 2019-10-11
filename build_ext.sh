#!/bin/sh

rm -rf build
rm mtuq/misfit/*.so

python ./setup.py build_ext --inplace

