#!/bin/bash

cd $(dirname ${BASH_SOURCE[0]})

rm examples/*.so
python ./setup.py build_ext --inplace

