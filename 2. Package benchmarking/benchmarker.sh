#!/bin/bash

lib=$1
round1=$2

# make and enter virtual-environment folder
mkdir ./ve
cd ve

# create virtual environment
virtualenv randomenv

# activate the virtual environment
source randomenv/bin/activate

# benchmarking
exec 3>&1 4>&2
#packsize_before=$(pip list | tail -n +3 | awk '{print $1}' | xargs pip show | grep -E 'Location:|Name:' | cut -d ' ' -f 2 | paste -d ' ' - - | awk '{print $2 "/" tolower($1)}' | xargs du -sh 2> /dev/null | sort -hr)
timetaken=$( { time pip install $lib --no-cache-dir 1>&3 2>&4; } 2>&1 )

# look for dependencies only on 1st set
if [[ $round1 -gt 0 ]]
then
  dependencies=$(python -m pip show $lib)
  packsize_after=$(pip list | tail -n +3 | awk '{print $1}' | xargs pip show | grep -E 'Location:|Name:' | cut -d ' ' -f 2 | paste -d ' ' - - | awk '{print $2 "/" tolower($1)}' | xargs du -sh 2> /dev/null | sort -hr)
fi

exec 3>&- 4>&-

# deactivate the virtual environment
deactivate

# go back to home directory
cd ..

# remove the virtual environment folder and the pip cache (stored in .cache/pip)
rm -r ve
rm -r .cache

echo $timetaken > timetaken.txt
#echo $packsize_before

if [[ $round1 -gt 0 ]]
then
  echo $dependencies > dependencies.txt
  echo $packsize_after > packsize.txt
fi

exit

