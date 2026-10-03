python3 split.py 0
python3 train_nonparallel_concataveragemax.py
python3 predict_nonparallel_concataveragemax.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.0
mv predictions.csv predictions.csv.0


python3 split.py 1 
python3 train_nonparallel_concataveragemax.py
python3 predict_nonparallel_concataveragemax.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.1
mv predictions.csv predictions.csv.1

python3 split.py 2
python3 train_nonparallel_concataveragemax.py
python3 predict_nonparallel_concataveragemax.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.2
mv predictions.csv predictions.csv.2

python3 split.py 3
python3 train_nonparallel_concataveragemax.py
python3 predict_nonparallel_concataveragemax.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.3
mv predictions.csv predictions.csv.3

python3 split.py 4
python3 train_nonparallel_concataveragemax.py
python3 predict_nonparallel_concataveragemax.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.4
mv predictions.csv predictions.csv.4

