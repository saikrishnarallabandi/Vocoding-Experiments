#!/bin/bash
# Note the features are in reverse order when read in, and are reversed and put in the right order when written out.
PHONELIST=festival/clunits/phonenames
STATELIST=festival/clunits/statenames

coeffs_dir=festival/coeffs
ordered_coeffs_dir=festival/coeffs_ordered
feats_binary=festival/feats_binary
mkdir -p ${feats_binary}
mkdir -p ${ordered_coeffs_dir}

for i in festival/coeffs/*.feats;
    do
    echo "Making the features for " $i
    f=`basename $i|cut -d '.' -f1`
    tac $i > ${ordered_coeffs_dir}/${f}.lab
    done

for i in festival/coeffs/*.mcep;
   do
   echo "Making the mceps for " $i
   fname=$(basename "$i")
   tac $i | cut -d ' ' -f 2-> ${ordered_coeffs_dir}/${fname}
   done

