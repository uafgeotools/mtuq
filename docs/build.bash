#!/bin/bash

cd $(dirname ${BASH_SOURCE[0]})

rm -rf _build
rm -rf library/generated/*
make html

