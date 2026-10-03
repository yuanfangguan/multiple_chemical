import pandas as pd

# === Load prediction files ===
pred1 = pd.read_csv("predictions_xgboost.csv")
pred2 = pd.read_csv("predictions_lightgbm.csv")
pred3 = pd.read_csv("predictions_catboost.csv")  # 新增的第三个文件

# === Check that stimulus columns match ===
if not (all(pred1['stimulus'] == pred2['stimulus']) and all(pred1['stimulus'] == pred3['stimulus'])):
    raise ValueError("Stimulus order does not match between the files!")

# === Average predictions ===
final_pred = pd.DataFrame()
final_pred['stimulus'] = pred1['stimulus']

# Get all label columns (assume the same labels in all files)
label_columns = [col for col in pred1.columns if col != 'stimulus']

for label in label_columns:
    final_pred[label] = (pred1[label] + pred2[label] + pred3[label]) / 3.0
    # Optional: Clip values to 0–5
    final_pred[label] = final_pred[label].clip(lower=0, upper=5)

# === Save averaged predictions ===
final_pred.to_csv("predictions.csv", index=False)
print("✅ Averaged predictions saved to predictions.csv")

