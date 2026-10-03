import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.AtomPairs import Pairs, Torsions

# Read file and filter
df = pd.read_csv("../raw/CID.csv")
df = df[df["is_solvent"] == "FALSE"].copy()
df = df[df["SMILES"].notna()].copy()
df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
df = df[df["mol"].notna()].copy()

# =========================
# 1. Morgan Count Fingerprint
# =========================
def get_morgan_count_fp(mol, radius=2, nBits=2048):
    fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius, nBits)
    arr = [0] * nBits
    for bit_id, count in fp.GetNonzeroElements().items():
        arr[bit_id % nBits] = count
    return arr

morgan_count = df["mol"].apply(get_morgan_count_fp)
morgan_count_df = pd.DataFrame(morgan_count.tolist())
morgan_count_df.insert(0, "molecule", df["molecule"].values)
morgan_count_df.to_csv("features_morgan_count.csv", index=False)

# =========================
# 3. Topological Torsion Fingerprint
# =========================
def get_torsion_fp(mol, nBits=2048):
    fp = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)
    arr = [0] * nBits
    for bit_id, count in fp.GetNonzeroElements().items():
        arr[bit_id % nBits] = count
    return arr


torsion = df["mol"].apply(get_torsion_fp)
torsion_df = pd.DataFrame(torsion.tolist())
torsion_df.insert(0, "molecule", df["molecule"].values)
torsion_df.to_csv("features_torsion.csv", index=False)

# =========================
# 4. Layered Fingerprint
# =========================
from rdkit.Chem import LayeredFingerprint

def get_layered_fp(mol, fpSize=2048):
    fp = LayeredFingerprint(mol, fpSize=fpSize)
    return list(fp)



layered = df["mol"].apply(get_layered_fp)
layered_df = pd.DataFrame(layered.tolist())
layered_df.insert(0, "molecule", df["molecule"].values)
layered_df.to_csv("features_layered.csv", index=False)

# =========================
# 5. Extended Physicochemical Descriptors
# =========================
def get_more_descriptors(mol):
    return pd.Series([
        Descriptors.MolWt(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.RingCount(mol),
        Descriptors.NumAromaticRings(mol)
    ])

desc_more_df = df["mol"].apply(get_more_descriptors)
desc_more_df.columns = [
    "MolWt", "HDonors", "HAcceptors", "LogP", "TPSA",
    "RotatableBonds", "FractionCSP3", "RingCount", "AromaticRings"
]
desc_more_df.insert(0, "molecule", df["molecule"].values)
desc_more_df.to_csv("features_descriptors_more.csv", index=False)

print("✅ Additional RDKit-based fingerprint and descriptor files saved:")
print("   - features_morgan_count.csv")
print("   - features_atom_pair.csv")
print("   - features_torsion.csv")
print("   - features_layered.csv")
print("   - features_descriptors_more.csv")

