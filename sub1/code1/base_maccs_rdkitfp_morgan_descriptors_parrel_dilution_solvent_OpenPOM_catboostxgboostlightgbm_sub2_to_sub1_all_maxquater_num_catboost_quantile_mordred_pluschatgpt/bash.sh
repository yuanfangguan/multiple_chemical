python3 split_train.py 0
python3 split.py 0
python3 train_catboost.py
python3 predict_catboost.py
python3 train_lightgbm.py
python3 predict_lightgbm.py
python3 train_xgboost.py
python3 train_xgboost.py
python3 average_ori.py
python3 train_sub2.py
python3 predict_sub2.py
python3 weight.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.0

python3 split.py 1
python3 train_ori.py
python3 predict_ori.py
python3 train_sub2.py
python3 predict_sub2.py
python3 weight.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.1

python3 split.py 2
python3 train_ori.py
python3 predict_ori.py
python3 train_sub2.py
python3 predict_sub2.py
python3 weight.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.2

python3 split.py 3
python3 train_ori.py
python3 predict_ori.py
python3 train_sub2.py
python3 predict_sub2.py
python3 weight.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.3

python3 split.py 4
python3 train_ori.py
python3 predict_ori.py
python3 train_sub2.py
python3 predict_sub2.py
python3 weight.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.4

