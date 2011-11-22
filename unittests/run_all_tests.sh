#!/bin/bash
#
# Run all the unit tests in the current directory
#

echo "Running unit tests with python 2"
for test in test_*.py
do
    echo "Now running '$test'"
    python $test
    #python3.2 $test
done

echo "Running unit tests with python 3"
for test in test_*.py
do
    echo "Now running '$test'"
    python3.2 $test
done
