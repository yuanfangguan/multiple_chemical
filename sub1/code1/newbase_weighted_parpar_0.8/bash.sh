python3 split_train.py 0

python3 split.py 6
python3 train_ori.py
python3 predict_ori.py
python3 train_nonparr_catboost.py
python3 predict_nonparr_catboost.py
python3 train_HLnomodred.py
python3 predict_HLnomodred.py
python3 train_nonparralel_HL.py
python3 predict_nonparralel_HL.py
python3 train_sub2.py
python3 predict_sub2.py
python3 weight.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.6

