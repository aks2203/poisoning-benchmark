#! /bin/bash

############################################################
#
# benchmark_all.sh
# Code to execute benchmark tests
# Developed as part of Poison Attack Benchmarking project
# June 2020
#
############################################################

if (( $# == 3 )); then
  for filepath in $1*/; do
    python benchmark_test.py --poisons_path ${filepath} --dataset $2  --$3 --output benchmark_100_trials
  done
elif (( $# == 2 )); then
  for filepath in $1*/; do
    python benchmark_test.py --poisons_path ${filepath} --dataset $2 --output benchmark_100_trials
  done
else
  echo "Illegal number of arguments."
fi
