#!/usr/bin/env python3
"""
Generate GNN features (Chemprop embeddings) for molecules in a CSV.

Input CSV schema (at minimum):
  - molecule : your ID (e.g., 10313079)
  - SMILES   : canonical or any valid SMILES
  - is_solvent : can be TRUE/FALSE or strings like "TRUE (Paraffin Oil)"

Output:
  - molecule_gnn_features.csv  (columns: molecule, fp_0, fp_1, ...)

Usage (defaults match your example paths):
  python gnn_features.py \
    --input ../raw/CID.csv \
    --output molecule_gnn_features.csv \
    --model /path/to/chemprop_model.ckpt \
    --smiles_col SMILES --id_col molecule --solvent_col is_solvent

Notes:
  * You must provide a trained Chemprop checkpoint (.ckpt) via --model.
  * If you have multiple checkpoints, you can repeat --model to ensemble.
  * Embedding layer can be chosen with --ffn_block_index (default: -1 penultimate).
"""

import argparse
import os
import sys
import tempfile
import subprocess
import shutil
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="../raw/CID.csv", help="Path to input CSV")
    p.add_argument("--output", default="molecule_gnn_features.csv", help="Path to output CSV")
    p.add_argument("--model", required=True, nargs="+", help="One or more Chemprop .ckpt paths")
    p.add_argument("--smiles_col", default="SMILES", help="Column name containing SMILES")
    p.add_argument("--id_col", default="molecule", help="Column name for molecule IDs")
    p.add_argument("--solvent_col", default="is_solvent", help="Column marking solvents (TRUE/False or 'TRUE (...)')")
    p.add_argument("--ffn_block_index", type=int, default=-1,
                   help="Chemprop FFN block index to export (e.g., -1 penultimate, 0 first block)")
    p.add_argument("--batch_size", type=int, default=512, help="Batch size for fingerprint export")
    p.add_argument("--num_workers", type=int, default=0, help="Dataloader workers for Chemprop")
    p.add_argument("--chemprop_bin", default="chemprop", help="Chemprop CLI executable name")
    return p.parse_args()

def check_chemprop(chemprop_bin):
    try:
        subprocess.run([chemprop_bin, "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError:
        sys.exit(
            "ERROR: 'chemprop' CLI not found.\n"
            "Install with:  pip install chemprop rdkit-pypi pandas\n"
            "Then re-run this script."
        )

def load_and_clean(input_path, smiles_col, id_col, solvent_col):
    df = pd.read_csv(input_path)
    if smiles_col not in df.columns or id_col not in df.columns:
        sys.exit(f"ERROR: CSV must contain columns '{id_col}' and '{smiles_col}'. Found: {list(df.columns)}")

    # Filter solvents if solvent_col exists
    if solvent_col in df.columns:
        # Treat values starting with 'TRUE' (e.g., 'TRUE (Paraffin Oil)') as True
        mask = df[solvent_col].astype(str).str.strip().str.upper().str.startswith("TRUE")
        df = df.loc[~mask].copy()

    # Drop rows with empty/NaN SMILES
    df[smiles_col] = df[smiles_col].astype(str)
    df = df[df[smiles_col].str.len() > 0]
    df = df[df[smiles_col].str.lower() != "nan"]

    # Drop exact duplicate SMILES to avoid duplicate embeddings
    df = df.drop_duplicates(subset=[smiles_col]).copy()

    if df.empty:
        sys.exit("ERROR: No valid (non-solvent, non-empty) SMILES left after cleaning.")

    return df[[id_col, smiles_col]].rename(columns={id_col: "molecule", smiles_col: "SMILES"})

def run_chemprop_fingerprint(chemprop_bin, smiles_csv, model_paths, out_csv, ffn_block_index, batch_size, num_workers):
    cmd = [
        chemprop_bin, "fingerprint",
        "--test-path", smiles_csv,
        "--smiles-columns", "SMILES",
        "--output", out_csv,
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--ffn-block-index", str(ffn_block_index)
    ]
    # Support multiple --model-path options for ensembling
    for mp in model_paths:
        if not os.path.isfile(mp):
            sys.exit(f"ERROR: model checkpoint not found: {mp}")
        cmd.extend(["--model-path", mp])

    print(">> Running:", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 or not os.path.exists(out_csv):
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        sys.exit("ERROR: Chemprop fingerprint export failed (see logs above).")

def merge_and_write(map_df, fps_csv, output_path):
    fps = pd.read_csv(fps_csv)
    if "SMILES" not in fps.columns:
        sys.exit("ERROR: Chemprop output missing 'SMILES' column.")

    # Merge on SMILES, then drop SMILES and put molecule first
    out = map_df.merge(fps, on="SMILES", how="inner").drop(columns=["SMILES"])

    # Ensure molecule is first
    cols = ["molecule"] + [c for c in out.columns if c != "molecule"]
    out = out[cols]

    # Sort by molecule for stability (optional)
    out = out.sort_values(by="molecule").reset_index(drop=True)

    # Rename feature columns to fp_0, fp_1, ...
    feat_cols = [c for c in out.columns if c != "molecule"]
    new_names = {c: f"fp_{i}" for i, c in enumerate(feat_cols)}
    out = out.rename(columns=new_names)

    out.to_csv(output_path, index=False)
    print(f">> Wrote {output_path} with shape {out.shape}")

def main():
    args = parse_args()
    check_chemprop(args.chemprop_bin)

    # 1) Load & clean
    df = load_and_clean(args.input, args.smiles_col, args.id_col, args.solvent_col)

    # 2) Create a temp folder and minimal SMILES CSV for Chemprop
    with tempfile.TemporaryDirectory() as tmpdir:
        smiles_csv = os.path.join(tmpdir, "smiles.csv")
        fps_csv = os.path.join(tmpdir, "chemprop_fps.csv")

        # Save SMILES plus an index to preserve mapping
        df[["SMILES"]].to_csv(smiles_csv, index=False)

        # Also keep (SMILES, molecule) map in memory (df) for merge later

        # 3) Run Chemprop fingerprint export
        run_chemprop_fingerprint(
            chemprop_bin=args.chemprop_bin,
            smiles_csv=smiles_csv,
            model_paths=args.model,
            out_csv=fps_csv,
            ffn_block_index=args.ffn_block_index,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # 4) Merge and write final table
        merge_and_write(df[["molecule", "SMILES"]], fps_csv, args.output)

if __name__ == "__main__":
    main()

