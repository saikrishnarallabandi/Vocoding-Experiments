import os
import  sys

wav_dir = sys.argv[1]
lab_dir = sys.argv[2]
story = sys.argv[3]

target_wav_dir = '/home/saikrishnalticmu/Projects/Blizzard_2018/data/wav'
target_lab_dir = '/home/saikrishnalticmu/Projects/Blizzard_2018/data/lab'

if not os.path.exists(target_lab_dir):
	os.makedirs(target_lab_dir)
	os.makedirs(target_wav_dir)

lab_files = sorted(os.listdir(lab_dir))
wav_files = sorted(os.listdir(wav_dir))


count = 0
for file in lab_files:
	start_time_array = []
	end_time_array = []
	sentence_array = []
	filename = file.split('.')[0]
	wavefile = wav_dir + '/' + filename + '.wav'
	f = open(lab_dir + '/' + file)
	for line in f:
		line = line.split('\n')[0].split('\t')
		start_time_array.append(line[0])
		end_time_array.append(line[1])
		sentence_array.append(line[2:])
	print "Sentence Array is: ", sentence_array	

	for (a,b,c) in zip(start_time_array, end_time_array, sentence_array):
		print "A,B,C are: ", a,b,c
		lab_fname = target_lab_dir + '/' + story + '_' + str(count+1).zfill(4) + '.lab'
                wav_fname = target_wav_dir + '/' + story + '_' + str(count+1).zfill(4) + '.wav' 
                count +=1
                cmd = 'sox ' + wavefile + ' ' + wav_fname + ' trim ' + str(a) + ' ' + str(float(b) - float(a))
                print cmd
                os.system(cmd)

                ch =' '.join(k for k in c)
                f = open(lab_fname, 'w')
                print "Character is : ", ch
                f.write(ch + '\n')