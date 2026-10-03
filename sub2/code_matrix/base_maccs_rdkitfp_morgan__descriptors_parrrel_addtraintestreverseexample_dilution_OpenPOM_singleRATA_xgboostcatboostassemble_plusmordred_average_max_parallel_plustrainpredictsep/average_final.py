import pandas as pd

# === Load prediction files ===
pred1 = pd.read_csv("predictions_ori.csv")
pred2 = pd.read_csv("predictions_sep.csv")  # 新增的第二个文件

# === Sort columns (excluding 'stimulus') ===
def sort_columns(df):
    cols = [col for col in df.columns if col != 'stimulus']
    sorted_cols = ['stimulus'] + sorted(cols)
    return df[sorted_cols]

pred1 = sort_columns(pred1)
pred2 = sort_columns(pred2)

# === Sort rows by stimulus ===
pred1 = pred1.sort_values(by='stimulus').reset_index(drop=True)
pred2 = pred2.sort_values(by='stimulus').reset_index(drop=True)

# === Check that stimulus columns match ===
if not (all(pred1['stimulus'] == pred2['stimulus'])):
    raise ValueError("Stimulus order does not match between the files!")

# === Average predictions ===
final_pred = pd.DataFrame()
final_pred['stimulus'] = pred1['stimulus']

# Get all label columns (now sorted and consistent)
label_columns = [col for col in pred1.columns if col != 'stimulus']

for label in label_columns:
    final_pred[label] = (pred1[label] * 0.8 + pred2[label] * 0.2)
    # Optional: Clip values to 0–5
    # final_pred[label] = final_pred[label].clip(lower=0, upper=5)

# === Save averaged predictions ===
final_pred.to_csv("predictions.csv", index=False)
print("✅ Averaged predictions saved to predictions.csv")

