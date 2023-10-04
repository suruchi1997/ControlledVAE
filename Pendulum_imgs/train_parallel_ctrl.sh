#!/bin/bash
rseeds=$(cat rseeds.txt)
betas=$(cat betas.txt)
# Save the original IFS value
OLDIFS=$IFS
IFS=','
for rseed in $rseeds; do
    for beta in $betas; do
        sbatch ./strain.sh $rseed $beta
    done
done
# Restore the original IFS value
IFS=$OLDIFS
