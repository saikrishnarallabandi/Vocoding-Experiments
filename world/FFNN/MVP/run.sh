exp_name='../expt_MVP_dynet'

mkdir -p $exp_name/logs $exp_name/models $exp_name/test $exp_name/resynth $exp_name/valid

# Train a small net
python map_small_dynet.py $exp_name

# Separate the coefficients
./do_synth synth_world $exp_name/resynth files.test
./do_synth synth_world $exp_name/test files.test


