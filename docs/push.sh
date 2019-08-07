#!/bin/sh

cd $(dirname ${BASH_SOURCE[0]})

rm -rf _build
rm -rf library/generated/*
make html

cd _build
git init
git remote add origin git@github.com:uafseismo/mtuq.git
git checkout -b gh-pages
git add -A
git commit -m "Added website"
git push -f origin gh-pages



