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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# === 强制只用一个 GPU（避免 peer mapping 资源耗尽）===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 可改为 "1", "2" 等

class SMILESIterableDataset(IterableDataset):
    def __init__(self, file_paths, tokenizer, max_length=512):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        for path in self.file_paths:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                if pd.notna(row['SMILES']) and pd.notna(row['Preferred Name']) and pd.notna(row['Systematic Name']):
                    prompt = f"Convert the following SMILES to names:\nSMILES: {row['SMILES']}\n"
                    completion = f"Preferred: {row['Preferred Name']}\nSystematic: {row['Systematic Name']}"
                    full_text = prompt + completion
                    tokenized = self.tokenizer(
                        full_text,
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_length,
                        return_tensors="pt"
                    )
                    tokenized["labels"] = tokenized["input_ids"].clone()
                    yield {key: val.squeeze(0) for key, val in tokenized.items()}


def main():
    model_path = "../../models/Llama-3.2-3B"
    data_dir = "all_SMILES"
    output_dir = "./llama3-smiles-checkpoints_500000"

    # === Load tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # === Load model (4-bit) and fix to single GPU ===
    current_gpu = torch.cuda.current_device()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": current_gpu},
        load_in_4bit=True,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # === Apply LoRA ===
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # === Build dataset ===
    file_paths = sorted(glob.glob(os.path.join(data_dir, "Compound_*.csv")))
    dataset = SMILESIterableDataset(file_paths, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # === Training config (with fixed fsdp) ===
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=500000,  # required for IterableDataset
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        no_cuda=False,
        local_rank=-1,
        fsdp=[],  # ✅ fix: must be list not None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # === Info ===
    print("Training on device:", model.device)
    print("Visible CUDA device count:", torch.cuda.device_count())

    trainer.train()


if __name__ == "__main__":
    main()

