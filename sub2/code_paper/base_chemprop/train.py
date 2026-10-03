#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Chemprop v2 training pipeline (no CLI args)
# Launch strategy:
#   1) Prefer console script 'chemprop' if found (shutil.which).
#   2) Otherwise try module entry points via `python -m`:
#        - chemprop.cli
#        - chemprop.command_line
#        - chemprop.train
#        - chemprop.bin.chemprop
#   3) Logs stdout/stderr for debugging.

import os
import sys
import subprocess
import shlex
import pandas as pd
import numpy as np
import shutil
from datetime import datetime

# =========================
# Hard-coded paths & params
# =========================
TRAIN_PATH = "train.csv"
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"
CID_SMILES_PATH = "../../data/raw/CID.csv"

PREPARED_TRAIN_CSV = "chemprop_train.csv"
SAVE_DIR = "models/chemprop_run"

# Targets (space-separated in v2; here as a Python list)
TARGET_COLUMNS = [
    "Intensity","Pleasantness","Green","Cucumber","Herbal","Mint","Woody","Pine","Floral","Powdery",
    "Fruity","Citrus","Tropical","Berry","Peach","Sweet","Caramellic","Vanilla","BrownSpice","Smoky",
    "Burnt","Roasted","Grainy","Meaty","Nutty","Fatty","Coconut","Waxy","Dairy","Buttery","Cheesy",
    "Sour","Fermented","Sulfurous","Garlic.Onion","Earthy","Mushroom","Musty","Ammonia","Fishy",
    "Fecal","Rotten.Decay","Rubber","Phenolic","Animal","Medicinal","Cooling","Sharp","Chlorine",
    "Alcoholic","Plastic","Ozone","Metallic"
]

# Training hyperparameters
EPOCHS = 40
BATCH_SIZE = 64
HIDDEN_SIZE = 1200         # not used unless you uncomment flag below
DEPTH = 3
ENSEMBLE_SIZE = 5
SEED = 42                  # not passed unless you uncomment flag below
FFN_NUM_LAYERS = 2
INIT_LR = 5e-4
SPLIT_TYPE_V2 = "SCAFFOLD_BALANCED"  # v2 value
METRIC = "rmse"                      # regression

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# Data prep
# =========================
def load_and_clean_cid_smiles(path):
    if not os.path.exists(path):
        sys.exit(f"ERROR: Missing file: {path}")
    df = pd.read_csv(path)
    required = {"molecule","SMILES","is_solvent"}
    if not required.issubset(df.columns):
        sys.exit(f"ERROR: {path} must have columns {required}, but has {set(df.columns)}")

    solvent_mask = df["is_solvent"].astype(str).str.strip().str.upper().str.startswith("TRUE")
    before = len(df)
    df = df.loc[~solvent_mask].copy()
    after = len(df)

    df["SMILES"] = df["SMILES"].astype(str)
    df = df[(df["SMILES"].str.len() > 0) & (df["SMILES"].str.lower() != "nan")]

    print(f"[CID] kept {len(df)}/{before} rows after removing solvents ({before-after} dropped as solvents)")
    return dict(zip(df["molecule"], df["SMILES"]))

def build_stimulus_to_smiles(stim_map_path, comp_map_path, cid_to_smiles):
    for p in (stim_map_path, comp_map_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: Missing file: {p}")

    stim = pd.read_csv(stim_map_path)
    comp = pd.read_csv(comp_map_path)

    if not {"id","components"}.issubset(stim.columns):
        sys.exit("ERROR: Stimulus map must have 'id' and 'components' columns.")
    if not {"id","CID"}.issubset(comp.columns):
        sys.exit("ERROR: Component map must have 'id' and 'CID' columns.")

    component_to_cid = dict(zip(comp["id"], comp["CID"]))

    def components_to_smi_list(components_str):
        out = []
        for tok in str(components_str).split(";"):
            tok = tok.strip()
            if tok.isdigit():
                cid = component_to_cid.get(int(tok))
                smi = cid_to_smiles.get(cid)
                if smi:
                    out.append(smi)
        return out

    stim_to_smi = {}
    for _, r in stim.iterrows():
        sid = r["id"]
        smi_list = components_to_smi_list(r.get("components",""))
        if smi_list:
            stim_to_smi[sid] = smi_list
    mapped = sum(len(v) for v in stim_to_smi.values())
    print(f"[MAP] stimuli with >=1 SMILES: {len(stim_to_smi)} ; total SMILES mapped: {mapped}")
    return stim_to_smi

def make_chemprop_training_csv(train_path, stim_to_smi, out_csv):
    if not os.path.exists(train_path):
        sys.exit(f"ERROR: Missing file: {train_path}")

    df = pd.read_csv(train_path)
    if "stimulus" not in df.columns:
        sys.exit("ERROR: train.csv must contain a 'stimulus' column.")

    missing = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: train.csv missing target columns: {missing}")

    rows = []
    dropped = 0
    for _, r in df.iterrows():
        sid = r["stimulus"]
        smi_list = stim_to_smi.get(sid, [])
        if smi_list:
            for smi in smi_list:
                row = {"smiles": smi}
                for t in TARGET_COLUMNS:
                    row[t] = r[t]
                rows.append(row)
        else:
            dropped += 1

    if not rows:
        sys.exit("ERROR: No rows to train on (no components mapped to SMILES).")

    df_cp = pd.DataFrame(rows)

    # Coerce to numeric; inspect for NaNs
    for t in TARGET_COLUMNS:
        df_cp[t] = pd.to_numeric(df_cp[t], errors="coerce")

    n_before = len(df_cp)
    rows_any_nan = df_cp[TARGET_COLUMNS].isna().any(axis=1).sum()
    print(f"[PREP] rows before dropna: {n_before} ; rows with any NaN: {rows_any_nan}")

    df_cp = df_cp.dropna(subset=TARGET_COLUMNS)
    df_cp = df_cp.drop_duplicates(subset=["smiles"] + TARGET_COLUMNS).reset_index(drop=True)
    n_after = len(df_cp)
    print(f"[PREP] rows after dropna + dedup: {n_after} (dropped {n_before - n_after}) ; stimuli with no mapping dropped: {dropped}")

    if n_after == 0:
        sys.exit("ERROR: After cleaning, no rows remain. Check target NaNs and mapping.")

    if not np.isfinite(df_cp[TARGET_COLUMNS].to_numpy()).all():
        sys.exit("ERROR: Non-finite values detected after coercion. Inspect your targets.")

    df_cp.to_csv(out_csv, index=False)
    print(f"Prepared Chemprop training CSV: {out_csv} (rows={len(df_cp)})")

# =========================
# Helpers
# =========================
def _run_logged(cmd, log_dir, label="chemprop"):
    os.makedirs(log_dir, exist_ok=True)
    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    print(">>", cmd_str)

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_path = os.path.join(log_dir, f"{label}_stdout_{ts}.log")
    stderr_path = os.path.join(log_dir, f"{label}_stderr_{ts}.log")

    with open(stdout_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout or "")
    with open(stderr_path, "w", encoding="utf-8") as f:
        f.write(proc.stderr or "")

    print(f"[LOG] stdout -> {stdout_path}")
    if proc.returncode != 0:
        print(f"[LOG] stderr -> {stderr_path}", file=sys.stderr)
        # Tail for quick view
        if proc.stderr:
            tail = "\n".join(proc.stderr.splitlines()[-20:])
            print("\n--- stderr (last 20 lines) ---", file=sys.stderr)
            print(tail, file=sys.stderr)
            print("--- end stderr tail ---", file=sys.stderr)

    return proc.returncode == 0, stdout_path, stderr_path

def _chemprop_cmd_candidates():
    """Return a list of (cmd_list, description) candidates to try, without importing them."""
    candidates = []

    # 1) Console script, if present on PATH
    bin_path = shutil.which("chemprop")
    if bin_path:
        candidates.append(([bin_path, "train"], f"console script: {bin_path}"))

    # 2) Module entry points (try running; don't pre-import)
    mod_entries = [
        "chemprop.cli",
        "chemprop.command_line",
        "chemprop.train",
        "chemprop.bin.chemprop",
    ]
    for mod in mod_entries:
        candidates.append(([sys.executable, "-m", mod, "train"], f"python -m {mod}"))

    return candidates

def build_train_args():
    base = [
        "--data-path", PREPARED_TRAIN_CSV,
        "--smiles-columns", "smiles",
        "--task-type", "regression",
        "--target-columns"
    ] + TARGET_COLUMNS + [
        "--save-dir", SAVE_DIR,
        "--split-type", SPLIT_TYPE_V2,
        "--metric", METRIC,
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--depth", str(DEPTH),
        # "--hidden-size", str(HIDDEN_SIZE),   # uncomment to control hidden size
        "--ensemble-size", str(ENSEMBLE_SIZE),
        # "--seed", str(SEED),                  # uncomment to fix seed
        "--ffn-num-layers", str(FFN_NUM_LAYERS),
        "--init-lr", str(INIT_LR),
        "--num-workers", "0",
    ]
    return base

def check_env_note():
    # Not fatal if chemprop isn't importable (some wheels install only the CLI)
    import platform
    pyver = platform.python_version()
    print(f"[ENV] Python {pyver}")
    if pyver.startswith("3.13"):
        print("⚠️  Python 3.13 detected. If you hit dependency issues, consider Python 3.10/3.11.", file=sys.stderr)

# =========================
# Chemprop runner (v2)
# =========================
def run_chemprop_flexible():
    log_dir = os.path.join(SAVE_DIR, "logs")
    args = build_train_args()

    tried = []
    for base_cmd, desc in _chemprop_cmd_candidates():
        cmd = base_cmd + args
        print(f">> Trying Chemprop via {desc}")
        ok, out_log, err_log = _run_logged(cmd, log_dir, label="chemprop")
        tried.append((desc, ok, out_log, err_log))
        if ok:
            print(f"✅ Success with {desc}")
            return True

    # If we get here, all candidates failed
    print("\n❌ All Chemprop invocation strategies failed.", file=sys.stderr)
    print("   What to try next:", file=sys.stderr)
    print("   • Ensure the console script is installed on PATH:", file=sys.stderr)
    print("       python -m pip show chemprop", file=sys.stderr)
    print("       which chemprop", file=sys.stderr)
    print("   • If no console script, reinstall v2 cleanly in THIS env:", file=sys.stderr)
    print("       python -m pip install --upgrade --force-reinstall 'chemprop>=2,<3'", file=sys.stderr)
    for desc, ok, out_log, err_log in tried:
        print(f"   - {desc}: {'OK' if ok else 'FAILED'} | logs: {out_log} , {err_log}", file=sys.stderr)
    return False

# =========================
# Main
# =========================
def main():
    check_env_note()

    # Build CID -> SMILES map
    cid_to_smiles = load_and_clean_cid_smiles(CID_SMILES_PATH)
    # Build stimulus -> [SMILES]
    stim_to_smi = build_stimulus_to_smiles(STIMULUS_MAP_PATH, COMPONENT_MAP_PATH, cid_to_smiles)
    # Create chemprop_train.csv
    make_chemprop_training_csv(TRAIN_PATH, stim_to_smi, PREPARED_TRAIN_CSV)

    # Quick sanity of prepared CSV
    if not os.path.exists(PREPARED_TRAIN_CSV):
        sys.exit(f"ERROR: Prepared CSV missing: {PREPARED_TRAIN_CSV}")
    df = pd.read_csv(PREPARED_TRAIN_CSV)
    print(f"[CHK] prepared rows: {len(df)} ; unique smiles: {df['smiles'].nunique()}")

    # Train via flexible launcher
    ok = run_chemprop_flexible()
    if not ok:
        sys.exit("ERROR: Chemprop training failed (see logs above).")
    print(f"✅ Chemprop model saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()

