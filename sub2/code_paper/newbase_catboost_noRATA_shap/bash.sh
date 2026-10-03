python3 split.py 0
python3 train_catboost.py
python3 predict_catboost.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.0
mv shap_outputs shap_outputs.0

python3 split.py 1
python3 train_catboost.py
python3 predict_catboost.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.1
mv shap_outputs shap_outputs.1

python3 split.py 2
python3 train_catboost.py
python3 predict_catboost.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.2
mv shap_outputs shap_outputs.2

python3 split.py 3
python3 train_catboost.py
python3 predict_catboost.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.3
mv shap_outputs shap_outputs.3

python3 split.py 4
python3 train_catboost.py
python3 predict_catboost.py
python3 eva.py
mv evaluation.tsv evaluation.tsv.4
mv shap_outputs shap_outputs.4

