import random

import sys
random.seed(int(sys.argv[1]))

TRAIN=open('train.csv','w')
TEST=open('test.csv','w')

REF=open('../../data/raw/TASK2_Train_mixture_Dataset.csv','r')
headline=REF.readline()
TRAIN.write(headline)
TEST.write(headline)
for line in REF:
    rrr=random.random()
    if (rrr<0.7):
        TRAIN.write(line)
    else:
        TEST.write(line)
REF.close()
TRAIN.close()
TEST.close()

