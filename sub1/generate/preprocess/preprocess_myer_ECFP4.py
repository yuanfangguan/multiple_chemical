import pandas as pd
from rdkit import Chem
import numpy as np
from molsig.Signature import MoleculeSignature

# ========================================
# Load CID file with SMILES
# ========================================
df = pd.read_csv("../../data/raw/CID.csv")
df = df[df["SMILES"].notna() & (df["SMILES"] != "")]

# ========================================
# molsig-compatible fingerprint generator
# ========================================
def molsig_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    ms = MoleculeSignature(
        mol,
        radius=radius,
        nbits=n_bits,
        map_root=True,
        use_stereo=True
    )

    # ms.morgans = list of lists (bit indices per atom)
    fp = np.zeros(n_bits, dtype=int)

    for atom_bits in ms.morgans:
        for bit in atom_bits:
            fp[int(bit)] += 1

    return fp

# ========================================
# Generate fingerprints
# ========================================
fingerprints = []
valid_rows = []

for _, row in df.iterrows():
    fp = molsig_fingerprint(row["SMILES"])
    if fp is not None:
        fingerprints.append(fp)
        valid_rows.append(row)

# ========================================
# Build dataframe
# ========================================
fp_df = pd.DataFrame(fingerprints, columns=[f"bit_{i}" for i in range(2048)])
meta_df = pd.DataFrame(valid_rows)[["molecule", "SMILES"]].reset_index(drop=True)
final_df = pd.concat([meta_df, fp_df], axis=1)

final_df.to_csv("cid_molsig_fp_2048.csv", index=False)
print("Saved cid_molsig_fp_2048.csv")
print("Shape:", final_df.shape)

