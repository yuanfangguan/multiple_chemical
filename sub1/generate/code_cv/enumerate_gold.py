import os
import numpy as np
import pandas as pd
from rdkit import Chem

from molsig.SignatureAlphabet import load_alphabet
from molsig.enumerate_signature import (
    enumerate_signature_from_morgan,
    enumerate_molecule_from_signature,
)

# ===========================
# 1. Load Alphabet
# ===========================
alphabet_path = "../alphabets/metanetx_alphabet.npz"
Alphabet = load_alphabet(alphabet_path, verbose=True)

print("Alphabet loaded.")
print(f"  radius = {Alphabet.radius}")
print(f"  nBits  = {Alphabet.nBits}")
print(f"  size   = {len(Alphabet.Dict)}")

# ===========================
# 2. Load gold fingerprints
# ===========================
df = pd.read_csv("../preprocess/cid_molsig_fp_2048.csv")
fp_cols = [f"bit_{i}" for i in range(2048)]

# ===========================
# 3. Output folder
# ===========================
out_dir = "enumerated_molecules"
os.makedirs(out_dir, exist_ok=True)

summary_rows = []

print("\n=== Starting full scan ===\n")

for idx in range(len(df)):

    row = df.iloc[idx]
    cid = row["molecule"]
    smi = row["SMILES"]
    fp = row[fp_cols].values.astype(int)

    print(f"\n[{idx}/{len(df)}] CID {cid}  SMILES={smi}")

    # Check fragment count
    mol = Chem.MolFromSmiles(smi)
    n_frag = len(Chem.GetMolFrags(mol)) if mol else -1

    # 1. Enumerate signatures
    Ssig, part_thresh, ct_morgan_sig, ct_solve = enumerate_signature_from_morgan(
        fp, Alphabet, max_nbr_partition=20000, verbose=False
    )

    sig_count = len(Ssig)
    print(f"  Signatures found: {sig_count}")

    mol_set = set()

    # 2. For each signature, enumerate molecules
    if sig_count > 0:
        for sig in Ssig[:10]:   # limit to first 10 signatures for speed
            Smol, rec_thresh, _ = enumerate_molecule_from_signature(
                sig,
                Alphabet,
                max_nbr_recursion=50000,
                repeat=3,
                clean_by_sig=True,
                verbose=False
            )
            mol_set |= set(Smol)

    mol_count = len(mol_set)
    print(f"  Molecules found: {mol_count}")

    # Save molecules to file
    if mol_count > 0:
        out_file = os.path.join(out_dir, f"mol_{idx}_{cid}.txt")
        with open(out_file, "w") as f:
            f.write(f"# CID {cid}\n")
            f.write(f"# SMILES {smi}\n")
            f.write("# Enumerated molecules:\n\n")
            for m in mol_set:
                f.write(m + "\n")
        print(f"  Saved molecules → {out_file}")

    # Add row to summary
    summary_rows.append({
        "idx": idx,
        "molecule": cid,
        "SMILES": smi,
        "n_frag": n_frag,
        "bits_on": int(fp.sum()),
        "signature_count": sig_count,
        "molecule_count": mol_count,
    })

# Save summary CSV
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("enumeration_scan_full_results.csv", index=False)

print("\n=== DONE ===")
print("Results saved to:")
print("  enumeration_scan_full_results.csv")
print("  enumerated_molecules/  (per-molecule SMILES files)")

