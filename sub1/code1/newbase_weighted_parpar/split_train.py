import random
import sys
random.seed(int(sys.argv[1]))

TRAIN=open('train_sub2.csv','w')

REF=open('../../../sub2/data/raw/TASK2_Train_mixture_Dataset.csv','r')
headline=REF.readline()
TRAIN.write(headline)
for line in REF:
    rrr=random.random()
    if (rrr<1.7):
        TRAIN.write(line)
REF.close()
TRAIN.close()

