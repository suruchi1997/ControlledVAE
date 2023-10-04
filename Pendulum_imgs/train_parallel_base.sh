#!/bin/bash
list=$(cat rseeds.txt)
# Save the original IFS value
OLDIFS=$IFS
IFS=','
for value in $list; do
    sbatch ./strain.sh $value 0.0
done
# Restore the original IFS value
IFS=$OLDIFS
