import random

import sys
random.seed(int(sys.argv[1]))

TRAIN=open('train_sub2tosub1.csv','w')
REF=open('../../../sub1/data/raw/TASK1_training.csv','r')
headline=REF.readline()
TRAIN.write(headline)
for line in REF:
    rrr=random.random()
    if (rrr<1.7):
        TRAIN.write(line)
REF.close()
TRAIN.close()

