import json
import os

file = json.load(open("json_extraction_dataset_500.json", "r"))
print(file[1])
# For GPU check
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
from unsloth import FastLanguageModel
import torch

model_name = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"

max_seq_length = 2048  # Choose sequence length
dtype = None  # Auto detection

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
)
from datasets import Dataset

def format_prompt(example):
    messages = [
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": json.dumps(example["output"])}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

formatted_data = [format_prompt(item) for item in file]
dataset = Dataset.from_dict({"text": formatted_data})
# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # LoRA rank - higher = more capacity, more memory
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=128,  # LoRA scaling factor (usually 2x rank)
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",     # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized version
    random_state=3407,
    use_rslora=False,  # Rank stabilized LoRA
    loftq_config=None, # LoftQ
)
from trl import SFTTrainer
from transformers import TrainingArguments

# Training arguments optimized for Unsloth
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 8
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=25,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_pin_memory=False,
    ),
)
# Train the model
trainer_stats = trainer.train()
# Test the fine-tuned model
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# Test prompt
messages = [
    {"role": "user", "content": "Extract the product information:\n<div class='product'><h2>iPad Air</h2><span class='price'>$1344</span><span class='category'>audio</span><span class='brand'>Dell</span></div>"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

# Generate response
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    use_cache=True,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
)

# Decode and print
response = tokenizer.batch_decode(outputs)[0]
print(response)
model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method="q4_k_m")
gguf_files = [f for f in os.listdir("gguf_model") if f.endswith(".gguf")]
if gguf_files:
    gguf_file = os.path.join("gguf_model", gguf_files[0])
    print(f"Downloading: {gguf_file}")
    import shutil
    shutil.copy(gguf_file, os.path.expanduser("~/Downloads/"))
    print(f"Saved to ~/Downloads/{gguf_files[0]}")