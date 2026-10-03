python3 split_train.py 0

python3 split_test.py 0
python3 train.py
python3 predict.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.0

python3 split_test.py 1
python3 train.py
python3 predict.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.1

python3 split_test.py 2
python3 train.py
python3 predict.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.2

python3 split_test.py 3
python3 train.py
python3 predict.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.3

python3 split_test.py 4
python3 train.py
python3 predict.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.4

