exp_name='../exp/basic_FFNNN'

mkdir -p $exp_name/logs

for hidden in 1024 512 #1024 
  do
   python  map_small.py  $hidden ${exp_name}
  done


