for file in feats_world/*_adjusted; do fname=$(basename "$file" .ccoeffs_ascii_adjusted); echo $fname; cp $file ss_dnn/data/output_full/$fname.ccoeffs; done

