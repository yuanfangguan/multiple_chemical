import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint, Descriptors
from mordred import Calculator, descriptors as mordred_descriptors
import numpy as np
import os

# ========================
# Paths
# ========================
input_file = "../data/drug.csv"
output_dir = "../data/features"
os.makedirs(output_dir, exist_ok=True)

# ========================
# 1. Read & Clean SMILES
# ========================
# Force all columns as strings to prevent .0 float conversion
df = pd.read_csv(input_file, dtype=str)

def clean_smiles(s):
    """Return first valid SMILES if multiple separated by ';'."""
    if pd.isna(s):
        return None
    for part in str(s).split(";"):
        part = part.strip()
        if part:
            mol = Chem.MolFromSmiles(part)
            if mol is not None:
                return part
    return None

df["clean_smiles"] = df["smiles"].apply(clean_smiles)
df["mol"] = df["clean_smiles"].apply(lambda s: Chem.MolFromSmiles(s) if pd.notna(s) else None)
df = df[df["mol"].notnull()].reset_index(drop=True)

print(f"✅ Loaded {len(df)} valid molecules from {input_file}")

# Use 'cid' or 'dname' as ID
id_col = "cid" if "cid" in df.columns else "dname"

# Ensure molecule IDs are strings (no .0 suffix)
df[id_col] = df[id_col].astype(str).str.replace(r"\.0$", "", regex=True)

# ========================
# 2. Morgan Fingerprint
# ========================
def get_morgan_fp(mol, radius=2, nBits=2048):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))

morgan = df["mol"].apply(get_morgan_fp)
morgan_df = pd.DataFrame(morgan.tolist())
morgan_df.insert(0, "molecule", df[id_col].values)
morgan_df.to_csv(os.path.join(output_dir, "features_morgan.csv"), index=False)
print("✅ Saved features_morgan.csv")

# ========================
# 3. MACCS Fingerprint
# ========================
def get_maccs_fp(mol):
    # Skip first bit (always 0)
    return list(MACCSkeys.GenMACCSKeys(mol))[1:]

maccs = df["mol"].apply(get_maccs_fp)
maccs_df = pd.DataFrame(maccs.tolist())
maccs_df.insert(0, "molecule", df[id_col].values)
maccs_df.to_csv(os.path.join(output_dir, "features_maccs.csv"), index=False)
print("✅ Saved features_maccs.csv")

# ========================
# 4. RDKit Fingerprint
# ========================
def get_rdkit_fp(mol, nBits=2048):
    return list(RDKFingerprint(mol, fpSize=nBits))

rdkit_fp = df["mol"].apply(get_rdkit_fp)
rdkit_fp_df = pd.DataFrame(rdkit_fp.tolist())
rdkit_fp_df.insert(0, "molecule", df[id_col].values)
rdkit_fp_df.to_csv(os.path.join(output_dir, "features_rdkitfp.csv"), index=False)
print("✅ Saved features_rdkitfp.csv")

# ========================
# 5. Physicochemical Descriptors
# ========================
def get_descriptors(mol):
    return pd.Series([
        Descriptors.MolWt(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol)
    ])

desc_df = df["mol"].apply(get_descriptors)
desc_df.columns = ["MolWt", "HDonors", "HAcceptors", "LogP", "TPSA"]
desc_df.insert(0, "molecule", df[id_col].values)
desc_df.to_csv(os.path.join(output_dir, "features_descriptors.csv"), index=False)
print("✅ Saved features_descriptors.csv")

# ========================
# 6. Mordred Descriptors
# ========================
print("Computing Mordred descriptors (this may take a while)...")
calc = Calculator(mordred_descriptors, ignore_3D=True)
mordred_df = calc.pandas(df["mol"])
mordred_df = mordred_df.replace([np.inf, -np.inf], np.nan).fillna(0)
mordred_df.insert(0, "molecule", df[id_col].values)
mordred_df.to_csv(os.path.join(output_dir, "features_mordred.csv"), index=False)
print("✅ Saved features_mordred.csv")

print("🎉 All feature files saved in", output_dir)

