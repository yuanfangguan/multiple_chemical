import pandas as pd
import glob
import json

output_file = "smiles2names.jsonl"
with open(output_file, "w") as out_f:
    for file in sorted(glob.glob("all_SMILES/Compound_*.csv")):
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            if pd.notna(row['SMILES']) and pd.notna(row['Preferred Name']) and pd.notna(row['Systematic Name']):
                prompt = f"Convert the following SMILES to names:\nSMILES: {row['SMILES']}\n"
                completion = f"Preferred: {row['Preferred Name']}\nSystematic: {row['Systematic Name']}"
                out_f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")

