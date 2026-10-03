#!/usr/bin/env python3
import argparse
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

def extract_kmers(seq: str, k: int) -> List[str]:
    n = len(seq)
    if n < k:
        return []
    return [seq[i:i+k] for i in range(n - k + 1)]

def count_kmers(seq: str, max_k: int = 4) -> Counter:
    seq = (seq or "").strip()
    counts = Counter()
    if not seq:
        return counts
    for k in range(1, max_k + 1):
        counts.update(extract_kmers(seq, k))
    return counts

def kmer_to_safe_col(k: int, kmer: str) -> str:
    """
    Make a LightGBM-safe column name:
      - Only [A-Za-z0-9_]
      - Encode the raw k-mer as hex
      - Prefix enc with 'k{K}_x'
    """
    hexmer = kmer.encode("utf-8").hex()  # 0-9a-f
    return f"k{k}_x{hexmer}"

def build_feature_frame(
    df: pd.DataFrame,
    max_k: int = 4,
    drop_missing_smiles: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      features_df: wide table with 'molecule' first, then k-mer count columns (hex-safe)
      mapping_df : rows of [feature, k, kmer] for decoding columns later
    """
    # Filter / fill SMILES
    if drop_missing_smiles:
        df = df[~df["SMILES"].isna() & (df["SMILES"].astype(str).str.strip() != "")]
    else:
        df = df.copy()
        df["SMILES"] = df["SMILES"].fillna("")

    # Build global vocab
    vocab_per_k: Dict[int, set] = defaultdict(set)
    for smiles in df["SMILES"].astype(str):
        s = smiles.strip()
        for k in range(1, max_k + 1):
            vocab_per_k[k].update(extract_kmers(s, k))

    # Order features: by k, then lexicographically by raw k-mer
    ordered: List[Tuple[int, str]] = []
    for k in range(1, max_k + 1):
        for mer in sorted(vocab_per_k[k]):
            ordered.append((k, mer))

    # Create mapping feature -> (k, kmer)
    feature_names: List[str] = []
    mapping_rows = []
    for k, mer in ordered:
        feat = kmer_to_safe_col(k, mer)
        feature_names.append(feat)
        mapping_rows.append({"feature": feat, "k": k, "kmer": mer})

    # Fill rows
    rows = []
    for _, r in df.iterrows():
        mol = r["molecule"]
        smiles = str(r["SMILES"]).strip()
        counts = count_kmers(smiles, max_k=max_k)

        feat_row = {"molecule": mol}
        # Use the same order as feature_names (deterministic)
        # We must lookup counts by raw kmer, so rebuild the raw kmer from mapping
        for m in mapping_rows:
            mer = m["kmer"]
            feat_row[m["feature"]] = counts.get(mer, 0)
        rows.append(feat_row)

    features_df = pd.DataFrame(rows, columns=["molecule"] + feature_names)
    mapping_df = pd.DataFrame(mapping_rows, columns=["feature", "k", "kmer"])
    return features_df, mapping_df

def main():
    ap = argparse.ArgumentParser(description="Create LightGBM-safe k-mer features (k=1..4) from SMILES.")
    ap.add_argument("--input", "-i", required=True, help="Input CSV with columns: molecule,SMILES,...")
    ap.add_argument("--output", "-o", required=True, help="Output feature CSV path.")
    ap.add_argument("--mapping", "-m", required=True, help="Output mapping CSV path.")
    ap.add_argument("--max_k", type=int, default=4, help="Max k for k-mers (default: 4).")
    ap.add_argument("--keep-missing", action="store_true",
                    help="Keep rows with missing/empty SMILES (counts will be zeros).")
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype={"molecule": object, "SMILES": object})
    if "molecule" not in df.columns or "SMILES" not in df.columns:
        raise ValueError("Input must contain columns 'molecule' and 'SMILES'.")

    features_df, mapping_df = build_feature_frame(
        df,
        max_k=args.max_k,
        drop_missing_smiles=not args.keep_missing
    )

    features_df.to_csv(args.output, index=False)
    mapping_df.to_csv(args.mapping, index=False)

    print(f"Wrote features: {args.output}  shape={features_df.shape}")
    print(f"Wrote mapping : {args.mapping}  rows={len(mapping_df)}")

if __name__ == "__main__":
    main()

