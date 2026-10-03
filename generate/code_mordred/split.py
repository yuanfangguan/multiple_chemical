import random

import sys
random.seed(int(sys.argv[1]))

TRAIN=open('train.csv','w')
TEST=open('test.csv','w')

REF=open('../../sub1/data/raw/TASK1_training.csv','r')
headline=REF.readline()
TRAIN.write(headline)
TEST.write(headline)
for line in REF:
    rrr=random.random()
    if (rrr<0.8):
        TRAIN.write(line)
    else:
        TEST.write(line)
REF.close()
TRAIN.close()
TEST.close()

