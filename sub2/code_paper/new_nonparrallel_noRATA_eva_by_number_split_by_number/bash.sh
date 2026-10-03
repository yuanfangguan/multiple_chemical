cp train_numcomp_2.csv train.csv
cp test_numcomp_2.csv test.csv
python3 train_nonparallel_concataveragemax.py
python3 predict_nonparallel_concataveragemax.py
python3 eva.py
mv evaluation_groups evaluation_groups_2

cp train_numcomp_3.csv train.csv
cp test_numcomp_3.csv test.csv
python3 train_nonparallel_concataveragemax.py
python3 predict_nonparallel_concataveragemax.py
python3 eva.py
mv evaluation_groups evaluation_groups_3

cp train_numcomp_5.csv train.csv
cp test_numcomp_5.csv test.csv
python3 train_nonparallel_concataveragemax.py
python3 predict_nonparallel_concataveragemax.py
python3 eva.py
mv evaluation_groups evaluation_groups_5

cp train_numcomp_10.csv train.csv
cp test_numcomp_10.csv test.csv
python3 train_nonparallel_concataveragemax.py
python3 predict_nonparallel_concataveragemax.py
python3 eva.py
mv evaluation_groups evaluation_groups_10

