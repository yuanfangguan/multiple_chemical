import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, RDKFingerprint

# Read file and filter
df = pd.read_csv("../raw/CID.csv")
df = df[df["is_solvent"] == "FALSE"].copy()
df = df[df["SMILES"].notna()].copy()
df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
df = df[df["mol"].notna()].copy()

# ========================
# 1. Morgan Fingerprint
# ========================
def get_morgan_fp(mol, radius=2, nBits=2048):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))

morgan = df["mol"].apply(get_morgan_fp)
morgan_df = pd.DataFrame(morgan.tolist())
morgan_df.insert(0, "molecule", df["molecule"].values)
morgan_df.to_csv("features_morgan.csv", index=False)

# ========================
# 2. MACCS Fingerprint
# ========================
def get_maccs_fp(mol):
    return list(MACCSkeys.GenMACCSKeys(mol))[1:]  # skip first bit

maccs = df["mol"].apply(get_maccs_fp)
maccs_df = pd.DataFrame(maccs.tolist())
maccs_df.insert(0, "molecule", df["molecule"].values)
maccs_df.to_csv("features_maccs.csv", index=False)

# ========================
# 3. RDKit Fingerprint
# ========================
def get_rdkit_fp(mol, nBits=2048):
    return list(RDKFingerprint(mol, fpSize=nBits))

rdkit_fp = df["mol"].apply(get_rdkit_fp)
rdkit_fp_df = pd.DataFrame(rdkit_fp.tolist())
rdkit_fp_df.insert(0, "molecule", df["molecule"].values)
rdkit_fp_df.to_csv("features_rdkitfp.csv", index=False)

# ========================
# 4. Physicochemical Descriptors
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
desc_df.insert(0, "molecule", df["molecule"].values)
desc_df.to_csv("features_descriptors.csv", index=False)

print("✅ RDKit-based fingerprint files saved:")
print("   - features_morgan.csv")
print("   - features_maccs.csv")
print("   - features_rdkitfp.csv")
print("   - features_descriptors.csv")

