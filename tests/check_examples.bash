#!/bin/bash -e


#
# Checks whether existing files match code generater output
#

FILENAMES="\
    ../examples/GridSearch.DoubleCouple.py\
    ../examples/SerialGridSearch.DoubleCouple.py\
    ../tests/benchmark_cap_mtuq.py\
    "

# navigate to mtuq/tests
cd $(dirname ${BASH_SOURCE[0]})

for filename in $FILENAMES
do
    cp ${filename}{,~}
done

python ../setup/code_generator.py
for filename in $FILENAMES
do
    echo "Checking $filename..."
    cmp ${filename}{,~}
    rm ${filename}~
done
echo "SUCCESS"
echo ""

