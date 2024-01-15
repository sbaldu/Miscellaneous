#!/bin/bash

# we want to check that all the image produced are in fact identical, as they should
arr=(./*.png)
for i in ${arr[@]}
do
  diff=$( diff -q ${arr[1]} $i )
  if [ "$diff" != "" ]
	then
		echo "Images ${arr[1]} and $i are different"
		exit 1
	fi
done
