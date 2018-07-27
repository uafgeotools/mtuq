#!/bin/bash -e


#
# Checks whether existing files match code generater output
#

FILENAMES="\
    ../examples/GridSearch.DoubleCouple.3Parameter.MPI.py\
    ../examples/GridSearch.DoubleCouple.3Parameter.Serial.py\
    ../tests/benchmark_cap_fk.py\
    "

# navigate to mtuq/tests
cd $(dirname ${BASH_SOURCE[0]})

for filename in $FILENAMES
do
    echo "Checking $filename..."
    cp ${filename}{,~}
done

python ../data/examples/code_generator.py
for filename in $FILENAMES
do
    cmp ${filename}{,~}
    rm ${filename}~
done
echo "SUCCESS"
echo ""

