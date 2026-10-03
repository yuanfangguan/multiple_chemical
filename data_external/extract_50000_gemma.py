import torch
import sys
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# === Paths ===
cid_csv_path = "../sub1/data/raw/CID.csv"
base_model_path = "../../models/gemma-3-4b"
lora_model_path="gemma-3-4b-checkpoints_50000/checkpoint-50000/"
output_csv_path = "gemma-3-4b-50000_cid_smiles_hidden_features_layer"+sys.argv[1]+".csv"

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# === Load base model and apply LoRA checkpoint ===
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, lora_model_path)
model.eval()

# === Load data ===
cid_df = pd.read_csv(cid_csv_path)
cid_df = cid_df[cid_df['SMILES'].notna()].reset_index(drop=True)

# === Collect features ===
features = []
for _, row in tqdm(cid_df.iterrows(), total=len(cid_df)):
    cid = row["molecule"]
    smiles = row["SMILES"]

    # Format prompt
    input_text = f"Convert the following SMILES to names:\nSMILES: {smiles}\n"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    ).to(model.device)

    # Forward pass to get hidden states
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

    # Get 5th layer hidden state (index 4)
    layer5_hidden = outputs.hidden_states[int(sys.argv[1])-1].squeeze(0).cpu()  # shape: (seq_len, hidden_dim)

    # Option 1: Mean pool all token embeddings
    mean_feature = layer5_hidden.mean(dim=0).numpy()  # shape: (hidden_dim,)

    # Save with CID and SMILES
    feature_row = {
        "molecule": cid,
        "SMILES": smiles,
        **{f"f{i}": val for i, val in enumerate(mean_feature)}
    }
    features.append(feature_row)

# === Save to CSV ===
features_df = pd.DataFrame(features)
features_df.to_csv(output_csv_path, index=False)

print(f"✅ Saved features to: {output_csv_path}")

