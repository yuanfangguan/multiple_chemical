import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Paths (edit as needed) ===
cid_csv_path = "../sub1/data/raw/CID.csv"
# Point to your *full fine-tune* checkpoint directory saved by Trainer:
ft_checkpoint_path = "./llama3-smiles-full-finetune_50000"
# argv[1] = layer number you want (1..num_layers). Example: "5"
output_csv_path = f"fullft_cid_smiles_hidden_features_layer{sys.argv[1]}.csv"

# === Load tokenizer ===
# If you saved tokenizer with your checkpoint, you can also do:
# tokenizer = AutoTokenizer.from_pretrained(ft_checkpoint_path, trust_remote_code=True)
# Otherwise, loading from the original base model path is fine too.
tokenizer = AutoTokenizer.from_pretrained(ft_checkpoint_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === Load full fine-tuned model ===
# BF16 if available, else FP16 (or fall back to float32 if needed)
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else torch.float16

model = AutoModelForCausalLM.from_pretrained(
    ft_checkpoint_path,
    device_map="auto",          # spreads across available GPUs if you have more than one
    torch_dtype=dtype,
    trust_remote_code=True
)
model.eval()

# === Load data ===
cid_df = pd.read_csv(cid_csv_path)
cid_df = cid_df[cid_df['SMILES'].notna()].reset_index(drop=True)

# === Helper: tokenize prompt ===
def build_inputs(smiles: str, max_length=512):
    text = f"Convert the following SMILES to names:\nSMILES: {smiles}\n"
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

# === Layer selection ===
# hidden_states layout: [0]=embeddings, [1]=block1, ..., [num_layers]=blockN
# If the user passes "5", we will take hidden_states[5] (the 5th transformer block).
try:
    layer_num = int(sys.argv[1])
    if layer_num < 1:
        raise ValueError
except Exception:
    raise ValueError("Please pass a valid layer number (1..num_layers), e.g. `python extract.py 5`")

features = []

with torch.no_grad():
    for _, row in tqdm(cid_df.iterrows(), total=len(cid_df)):
        cid = row["molecule"]
        smiles = row["SMILES"]

        inputs = build_inputs(smiles).to(model.device)

        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

        # Get the requested layer's hidden states (seq_len, hidden_dim)
        # Note: hidden_states[0] are embeddings; we want transformer block `layer_num`.
        if layer_num >= len(outputs.hidden_states):
            raise IndexError(
                f"Requested layer {layer_num} but model returned only {len(outputs.hidden_states)-1} layers "
                "(excluding embeddings)."
            )

        layer_hidden = outputs.hidden_states[layer_num].squeeze(0).float().cpu()

        # Mean pool across tokens -> (hidden_dim,)
        mean_feature = layer_hidden.mean(dim=0).numpy()

        feature_row = {
            "molecule": cid,
            "SMILES": smiles,
            **{f"f{i}": val for i, val in enumerate(mean_feature)}
        }
        features.append(feature_row)

# === Save ===
features_df = pd.DataFrame(features)
features_df.to_csv(output_csv_path, index=False)
print(f"✅ Saved features to: {output_csv_path}")

