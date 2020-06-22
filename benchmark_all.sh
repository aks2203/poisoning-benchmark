#! /bin/bash

############################################################
#
# benchmark_all.sh
# Code to execute benchmark tests
# Developed as part of Poison Attack Benchmarking project
# June 2020
#
############################################################

if (( $# == 2 )); then
  for filepath in $1*/; do
    python benchmark_test.py --poisons_path ${filepath} --$2 --output benchmark_100_trials
  done
elif (( $# == 1 )); then
  for filepath in $1*/; do
    python benchmark_test.py --poisons_path ${filepath} --output benchmark_100_trials
  done
else
  echo "Illegal number of arguments."
fi
