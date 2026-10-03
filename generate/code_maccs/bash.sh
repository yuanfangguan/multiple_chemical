python3 split.py 0
python3 train.py
python3 predict.py
python3 evaluate.py
mv evaluation_summary.csv evaluation_summary.csv.0

python3 split.py 1 
python3 train.py
python3 predict.py
python3 evaluate.py
mv evaluation_summary.csv evaluation_summary.csv.1

python3 split.py 2
python3 train.py
python3 predict.py
python3 evaluate.py
mv evaluation_summary.csv evaluation_summary.csv.2

python3 split.py 3
python3 train.py
python3 predict.py
python3 evaluate.py
mv evaluation_summary.csv evaluation_summary.csv.3

python3 split.py 4
python3 train.py
python3 predict.py
python3 evaluate.py
mv evaluation_summary.csv evaluation_summary.csv.4
