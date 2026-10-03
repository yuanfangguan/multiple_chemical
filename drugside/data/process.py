import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, RDKFingerprint
from mordred import Calculator, descriptors

# ============================================================
# 0. Load TWOSIDES_with_smiles.csv
# ============================================================
df = pd.read_csv("TWOSIDES_with_smiles.csv")

# Collect unique SMILES
smiles_all = pd.Series(
    pd.concat([df["drug_1_smiles"], df["drug_2_smiles"]]).unique()
)
smiles_all = smiles_all[smiles_all.notna()].reset_index(drop=True)

# Build DataFrame with SMILES + mol objects
chem_df = pd.DataFrame({"SMILES": smiles_all})
chem_df["mol"] = chem_df["SMILES"].apply(Chem.MolFromSmiles)
chem_df = chem_df[chem_df["mol"].notna()].copy()
chem_df["molecule_id"] = chem_df.index.astype(str)

# ============================================================
# 1. Morgan Fingerprint
# ============================================================
def get_morgan_fp(mol, radius=2, nBits=2048):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))

morgan = chem_df["mol"].apply(get_morgan_fp)
morgan_df = pd.DataFrame(morgan.tolist())
morgan_df.insert(0, "molecule_id", chem_df["molecule_id"].values)
morgan_df.insert(1, "SMILES", chem_df["SMILES"].values)
morgan_df.to_csv("features_morgan.csv", index=False)

# ============================================================
# 2. MACCS Fingerprint
# ============================================================
def get_maccs_fp(mol):
    return list(MACCSkeys.GenMACCSKeys(mol))[1:]  # skip first bit

maccs = chem_df["mol"].apply(get_maccs_fp)
maccs_df = pd.DataFrame(maccs.tolist())
maccs_df.insert(0, "molecule_id", chem_df["molecule_id"].values)
maccs_df.insert(1, "SMILES", chem_df["SMILES"].values)
maccs_df.to_csv("features_maccs.csv", index=False)

# ============================================================
# 3. RDKit Fingerprint
# ============================================================
def get_rdkit_fp(mol, nBits=2048):
    return list(RDKFingerprint(mol, fpSize=nBits))

rdkit_fp = chem_df["mol"].apply(get_rdkit_fp)
rdkit_fp_df = pd.DataFrame(rdkit_fp.tolist())
rdkit_fp_df.insert(0, "molecule_id", chem_df["molecule_id"].values)
rdkit_fp_df.insert(1, "SMILES", chem_df["SMILES"].values)
rdkit_fp_df.to_csv("features_rdkitfp.csv", index=False)

# ============================================================
# 4. Basic RDKit Descriptors
# ============================================================
def get_rdkit_desc(mol):
    return pd.Series([
        Descriptors.MolWt(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol)
    ])

desc = chem_df["mol"].apply(get_rdkit_desc)
desc.columns = ["MolWt", "HDonors", "HAcceptors", "LogP", "TPSA"]

desc.insert(0, "molecule_id", chem_df["molecule_id"].values)
desc.insert(1, "SMILES", chem_df["SMILES"].values)

desc.to_csv("features_descriptors.csv", index=False)

# ============================================================
# 5. Mordred Descriptors (most expensive step)
# ============================================================
calc = Calculator(descriptors, ignore_3D=True)
mordred_df = calc.pandas(chem_df["mol"])

# Attach IDs
mordred_df.insert(0, "molecule_id", chem_df["molecule_id"].values)
mordred_df.insert(1, "SMILES", chem_df["SMILES"].values)

mordred_df.to_csv("features_mordred.csv", index=False)

# ============================================================
# 6. Save molecule map for later merging
# ============================================================
chem_df[["molecule_id", "SMILES"]].to_csv("molecule_map.csv", index=False)

print("✅ All feature files written:")
print("   - features_morgan.csv")
print("   - features_maccs.csv")
print("   - features_rdkitfp.csv")
print("   - features_descriptors.csv")
print("   - features_mordred.csv")
print("   - molecule_map.csv")

