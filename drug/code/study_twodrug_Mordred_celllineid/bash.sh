rm -rf results*
python3 split.py 0
python3 train.py
mv results results_0

python3 split.py 1
python3 train.py
mv results results_1

python3 split.py 2
python3 train.py
mv results results_2

python3 split.py 3
python3 train.py
mv results results_3

python3 split.py 4
python3 train.py
mv results results_4
