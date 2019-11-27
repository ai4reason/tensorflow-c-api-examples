#!/bin/bash

export LD_LIBRARY_PATH=$PWD/../lib:$LD_LIBRARY_PATH

gcc -I../include -L../lib tfc-version.c -ltensorflow -o tfc-version
gcc -I../include -L../lib tfc-model-list.c -ltensorflow -o tfc-model-list
gcc -I../include -L../lib tfc-model-eval.c -ltensorflow -o tfc-model-eval

./tfc-version

