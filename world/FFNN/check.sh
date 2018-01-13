#!/bin/bash

for file in festival/coeffs/arctic_a00*.feats; 
 do 
     fname=$(basename "$file" .feats)  
     feat_length=`wc -l $file | cut -d " " -f 1`
     coeff_length=`wc -l feats_world/$fname.ccoeffs_ascii_adjusted | cut -d " " -f 1`
     if [ $feat_length != $coeff_length ] 
       then 
          echo $fname " failed. Features are of length " $feat_length " while coeffs are of length:  " $coeff_length
       fi
 done
