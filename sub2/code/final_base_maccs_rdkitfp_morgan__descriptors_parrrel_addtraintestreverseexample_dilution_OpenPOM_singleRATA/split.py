import random

import sys
random.seed(int(sys.argv[1]))

TRAIN=open('train.csv','w')

REF=open('../../data/raw/TASK2_Train_mixture_Dataset.csv','r')
headline=REF.readline()
TRAIN.write(headline)
for line in REF:
    rrr=random.random()
    if (rrr<1.7):
        TRAIN.write(line)
REF.close()
TRAIN.close()


TEST=open('test.csv','w')
FILE=open('../../data/raw/TASK2_Leaderboard_set_Submission_form.csv','r')
headline=FILE.readline()
TEST.write(headline)
for line in FILE:
    TEST.write(line)
TEST.close()

