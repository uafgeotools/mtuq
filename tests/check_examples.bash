#!/bin/bash -e


#
# Checks whether code generater output matches examples included in repository
#


# navigate to mtuq/tests
cd $(dirname ${BASH_SOURCE[0]})

for filename in \
    ../examples/GridSearch.DoubleCouple.3Parameter.MPI.py\
    ../examples/GridSearch.DoubleCouple.3Parameter.Serial.py\
    ../tests/benchmark_cap_fk.py;
do
    echo "Checking $filename..."
    cp ${filename}{,~}

    python ../data/examples/code_generator.py
    diff ${filename}{,~}

    rm ${filename}~
done
echo "SUCCESS"
echo ""

