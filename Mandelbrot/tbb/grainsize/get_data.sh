#!/bin/bash

rm -f data.csv
touch data.csv
echo -n "grainsize" >> data.csv

n_rep=10
for ((i = 0; i < $n_rep; i++))
do
  echo -n ',time_' >> data.csv
  echo -n $i >> data.csv
done
echo '' >> data.csv

echo "Collecting data"
for ((i = 1; i <= 800; i += 1))
do
  echo $i
  echo -n "$i" >> data.csv

  for ((j = 0; j < $n_rep; j++))
  do
	echo -n ',' >> data.csv
	../../mandelbrot_tbb_release.out $i $i | tr -d "\n" >> data.csv
  done
  echo '' >> data.csv
done
