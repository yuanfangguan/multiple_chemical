import os
import glob
import pandas as pd
import torch
from torch.utils.data import IterableDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# === Force a single GPU (adjust if needed) ===
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

# Optional: small speedup on Ampere+/Ada
try:
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass


class SMILESIterableDataset(IterableDataset):
    """
    Streams rows from Compound_*.csv files and returns tokenized examples.
    The loss is computed only on the completion (labels for prompt tokens are -100).
    """
    def __init__(self, file_paths, tokenizer, max_length=512):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        for path in self.file_paths:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                if pd.notna(row.get('SMILES')) and pd.notna(row.get('Preferred Name')) and pd.notna(row.get('Systematic Name')):
                    prompt = (
                        "Convert the following SMILES to names:\n"
                        f"SMILES: {row['SMILES']}\n"
                    )
                    completion = (
                        f"Preferred: {row['Preferred Name']}\n"
                        f"Systematic: {row['Systematic Name']}"
                    )
                    full_text = prompt + completion

                    tokenized = self.tokenizer(
                        full_text,
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_length,
                        return_tensors="pt"
                    )

                    # --- Mask prompt tokens from loss ---
                    with self.tokenizer.as_target_tokenizer():
                        prompt_tok = self.tokenizer(
                            prompt,
                            truncation=True,
                            padding=False,
                            max_length=self.max_length,
                            return_tensors="pt"
                        )
                    labels = tokenized["input_ids"].clone()
                    prompt_len = prompt_tok.input_ids.shape[1]
                    # only mask within max_length range
                    mask_len = min(prompt_len, labels.shape[1])
                    labels[0, :mask_len] = -100

                    tokenized["labels"] = labels
                    yield {k: v.squeeze(0) for k, v in tokenized.items()}


def main():
    # === Paths ===
    model_path = "../../models/Llama-3.2-3B"
    data_dir = "../data_external/all_SMILES"
    output_dir = "./llama3-smiles-full-finetune_50000"

    # === Device & dtype selection ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # === Tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Ensure pad token exists and aligns config
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === Load full model ===
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device)

    # Gradient checkpointing & cache
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False  # required when gradient checkpointing is on
    # improves grad flow with checkpointing
    try:
        model.enable_input_require_grads = True
    except Exception:
        pass

    # If falling back to FP16 path, force params to half to avoid AMP unscale errors
    bf16_flag = use_bf16
    fp16_flag = not use_bf16
    if fp16_flag:
        # Make sure ALL params are FP16 so GradScaler doesn't see mixed dtypes
        model.half()

    # === Dataset & Collator ===
    file_paths = sorted(glob.glob(os.path.join(data_dir, "Compound_*.csv")))
    if not file_paths:
        raise FileNotFoundError(f"No files matched {os.path.join(data_dir, 'Compound_*.csv')}")

    dataset = SMILESIterableDataset(file_paths, tokenizer, max_length=512)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # === Training Args ===
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=50000,                 # IterableDataset requires max_steps
        learning_rate=2e-5,              # smaller LR for full FT vs LoRA
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        no_cuda=False,
        gradient_checkpointing=True,
        # precision flags
        bf16=bf16_flag,
        fp16=fp16_flag,
        # optimizer/schedule
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        # keep FSDP off unless you intend to use it
        fsdp=[],
    )

    # === Trainer ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        # Future-proof deprecation: tokenizer -> processing_class
        processing_class=tokenizer,
    )

    # === Info ===
    print("Training on device:", device)
    print("Visible CUDA device count:", torch.cuda.device_count())
    print("Precision:", "BF16" if bf16_flag else "FP16")

    trainer.train()


if __name__ == "__main__":
    main()

