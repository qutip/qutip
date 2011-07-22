#!/bin/bash
#
# Run all the unit tests in the current directory
#

for test in test_*.py
do
    echo "Now running '$test'"
    python $test
done
