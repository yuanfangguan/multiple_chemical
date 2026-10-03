python3 split.py 0
python3 train_nonparallel_concataveragemax.py
python3 predict_nonparallel_concataveragemax.py
python3 eva.py
python3 pick.py
mv evaluation.tsv evaluation.tsv.0
mv smells_detailed.csv smells_detailed.csv.0

python3 split.py 1 
python3 train_nonparallel_concataveragemax.py
python3 predict_nonparallel_concataveragemax.py
python3 eva.py
python3 pick.py
mv evaluation.tsv evaluation.tsv.1
mv smells_detailed.csv smells_detailed.csv.1

python3 split.py 2
python3 train_nonparallel_concataveragemax.py
python3 predict_nonparallel_concataveragemax.py
python3 eva.py
python3 pick.py
mv evaluation.tsv evaluation.tsv.2
mv smells_detailed.csv smells_detailed.csv.2

python3 split.py 3
python3 train_nonparallel_concataveragemax.py
python3 predict_nonparallel_concataveragemax.py
python3 eva.py
python3 pick.py
mv evaluation.tsv evaluation.tsv.3
mv smells_detailed.csv smells_detailed.csv.3

python3 split.py 4
python3 train_nonparallel_concataveragemax.py
python3 predict_nonparallel_concataveragemax.py
python3 eva.py
python3 pick.py
mv evaluation.tsv evaluation.tsv.4
mv smells_detailed.csv smells_detailed.csv.4

