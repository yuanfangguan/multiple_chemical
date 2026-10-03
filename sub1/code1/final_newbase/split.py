import random

import sys
random.seed(int(sys.argv[1]))

TRAIN=open('train.csv','w')

REF=open('../../data/raw/TASK1_training.csv','r')
headline=REF.readline()
TRAIN.write(headline)
for line in REF:
    rrr=random.random()
    TRAIN.write(line)
REF.close()
TRAIN.close()

TEST=open('test.csv','w')
FILE=open('../../data/raw/TASK1_leaderboard_set_Submission_form.csv','r')
for line in FILE:
    TEST.write(line)
TEST.close()
