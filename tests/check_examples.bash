#!/bin/bash -e


#
# Checks whether existing files match code generater output
#

FILENAMES="\
    ../examples/GridSearch.DoubleCouple.py\
    ../examples/GridSearch.DoubleCouple+Magnitude+Depth.py\
    ../examples/GridSearch.FullMomentTensor.py\
    ../examples/SerialGridSearch.DoubleCouple.py\
    ../setup/chinook/examples/CapStyleGridSearch.DoubleCouple.py\
    ../setup/chinook/examples/CapStyleGridSearch.DoubleCouple+Magnitude+Depth.py\
    ../tests/benchmark_cap.py\
    ../tests/test_graphics.py\
    ../tests/test_grid_search_mt.py\
    ../tests/test_grid_search_mt_depth.py\
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

