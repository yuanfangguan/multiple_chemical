import pandas as pd

# === Load prediction files ===
pred1 = pd.read_csv("predictions_xgboost.csv")
pred2 = pd.read_csv("predictions_lightgbm.csv")

# === Check that stimulus columns match ===
if not all(pred1['stimulus'] == pred2['stimulus']):
    raise ValueError("Stimulus order does not match between the two files!")

# === Average predictions ===
final_pred = pd.DataFrame()
final_pred['stimulus'] = pred1['stimulus']

# Get all label columns (assume the same labels in both files)
label_columns = [col for col in pred1.columns if col != 'stimulus']

for label in label_columns:
    final_pred[label] = (pred1[label] + pred2[label]) / 2.0
    # Optional: If you want to clip to 0-5 as before
    final_pred[label] = final_pred[label].clip(lower=0, upper=5)

# === Save averaged predictions ===
final_pred.to_csv("predictions.csv", index=False)
print("✅ Averaged predictions saved to predictions.csv")

