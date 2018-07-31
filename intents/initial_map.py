import os
import numpy as np
from keras.utils import to_categorical
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

phones_file = '../voices/cmu_us_awb_arctic/ehmm/etc/txt.phseq.data'
ccoeffs_folder = '../feats/awb_5msec'
ccoeffs_files = sorted(os.listdir(ccoeffs_folder))
p = defaultdict(int)


class arctic_dataset(Dataset):
      
      def __init__(self, ccoeffs_folder, phones_file):
        self.phones_file = phones_file
        self.ccoeffs_folder = ccoeffs_folder
        self.phones_array = []
        self.ccoeffs_array = []
        self.filenames_array = []
        f = open(self.phones_file)
        phones_dict = {}
        for line in f:
           line = line.split('\n')[0].split()
           fname = line[0]
           phones = line[1:]
           phones_numeric = []
           for phone in phones:
              phones_numeric.append(p[phone] +1)
           phones_onehot = to_categorical(phones_numeric, 43)
           ccoeffs = np.loadtxt(ccoeffs_folder + '/' + fname + '.ccoeffs_ascii')
           self.phones_array.append(phones_onehot)
           self.filenames_array.append(fname)
           self.ccoeffs_array.append(ccoeffs)

      def __getitem__(self, index):
           return self.filenames_array[index], self.phones_array[index], self.ccoeffs_array[index]


      def __len__(self):
           ccoeffs_files = sorted(os.listdir(self.ccoeffs_folder)) 
           return len(ccoeffs_files) 



wks_train = arctic_dataset(ccoeffs_folder, phones_file)
train_loader = DataLoader(wks_train,
                          batch_size=1,
                          shuffle=True,
                          num_workers=4
                         )


for (a,b,c) in train_loader:
     print (a)
