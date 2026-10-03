python3 split.py 0
python3 train_nonparallel_concataveragemax.py
python3 predict_nonparallel_concataveragemax.py
python3 eva.py
python3 pick_suppressed.py
python3 pick_emergent.py 
mv evaluation.tsv evaluation.tsv.0
mv suppressed_smells_detailed.csv suppressed_smells_detailed.csv.0
mv synergistic_smells_detailed.csv synergistic_smells_detailed.csv.0

python3 split.py 1 
python3 train_nonparallel_concataveragemax.py
python3 predict_nonparallel_concataveragemax.py
python3 eva.py
python3 pick_suppressed.py
python3 pick_emergent.py 
mv evaluation.tsv evaluation.tsv.1
mv suppressed_smells_detailed.csv suppressed_smells_detailed.csv.1
mv synergistic_smells_detailed.csv synergistic_smells_detailed.csv.1

python3 split.py 2
python3 train_nonparallel_concataveragemax.py
python3 predict_nonparallel_concataveragemax.py
python3 eva.py
python3 pick_suppressed.py
python3 pick_emergent.py 
mv evaluation.tsv evaluation.tsv.2
mv suppressed_smells_detailed.csv suppressed_smells_detailed.csv.2
mv synergistic_smells_detailed.csv synergistic_smells_detailed.csv.2

python3 split.py 3
python3 train_nonparallel_concataveragemax.py
python3 predict_nonparallel_concataveragemax.py
python3 eva.py
python3 pick_suppressed.py
python3 pick_emergent.py 
mv evaluation.tsv evaluation.tsv.3
mv suppressed_smells_detailed.csv suppressed_smells_detailed.csv.3
mv synergistic_smells_detailed.csv synergistic_smells_detailed.csv.3

python3 split.py 4
python3 train_nonparallel_concataveragemax.py
python3 predict_nonparallel_concataveragemax.py
python3 eva.py
python3 pick_suppressed.py
python3 pick_emergent.py 
mv evaluation.tsv evaluation.tsv.4
mv suppressed_smells_detailed.csv suppressed_smells_detailed.csv.4
mv synergistic_smells_detailed.csv synergistic_smells_detailed.csv.4

