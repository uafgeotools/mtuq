#!/bin/bash -e


function download() {

    # where data are hosted remotely
    remote="https://github.com/uafgeotools/mtuq.git"
    branch=$2

    # where data will be downloaded locally
    dirname=$1
    basename=$2
    fullname=${dirname}/${basename}

    echo "git clone --branch $branch $remote $fullname"
    git clone --branch $branch $remote $fullname

    cat ${fullname}/part* > ${fullname}.tgz
    rm ${fullname}/part*

    cd $dirname
    tar -xzf ${fullname}.tgz
    }


# directory in which download.bash resides
cd $(dirname ${BASH_SOURCE[0]})
wd=$(pwd)


for testcase in \
    benchmark_cap\
    benchmark_cps;
do
    cd $wd
    download $wd $testcase
done

