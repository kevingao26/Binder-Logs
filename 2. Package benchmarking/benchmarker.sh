#!/bin/bash

lib=$1

# make and enter virtual-environment folder
mkdir ./ve
cd ve

# create virtual environment
virtualenv randomenv

# activate the virtual environment
source randomenv/bin/activate

# benchmarking
exec 3>&1 4>&2
timetaken=$( { time pip install $lib --no-cache-dir 1>&3 2>&4; } 2>&1 )
exec 3>&- 4>&-

# deactivate the virtual environment
deactivate

# go back to home directory
cd ..

# remove the virtual environment folder and the pip cache (stored in .cache/pip)
rm -r ve
rm -r .cache

echo $timetaken > timetaken.txt
exit
