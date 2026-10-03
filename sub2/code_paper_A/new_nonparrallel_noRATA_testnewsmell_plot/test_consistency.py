import sys

FILE=open(sys.argv[1],'r')
FILE.readline()

count_correct=0
count_wrong=0
for line in FILE:
    line=line.strip()
    table=line.split(',')
    a=float(table[2])-float(table[4]) 
    b=float(table[3])-float(table[4]) 
    c=a*b
    if (c>0):
        count_correct=count_correct+1
    else:

        count_wrong=count_wrong+1
print(count_correct,count_wrong)
    
