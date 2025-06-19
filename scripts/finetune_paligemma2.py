#!/usr/bin/env python
# coding: utf-8

# Fine-Tuning PaliGemma-2 for Visual Pointing Tasks on Multiple GPUs

import os
import random
from pathlib import Path
import warnings
from datetime import datetime
from functools import partial

import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig, TrainingArguments, Trainer

from utils import check_image, download_and_cache_image, points_to_text

# --- Multi-GPU Setup Note ---
# The following two assertions ensure that a CUDA-enabled GPU environment is available
# and that PyTorch can see more than one GPU. If you want to train on specific GPUs,
# set the environment variable in your terminal before running the script.
# Example for using GPUs 0 and 1:
# export CUDA_VISIBLE_DEVICES=0,1
assert torch.cuda.is_available(), "CUDA is not available. Multi-GPU training requires a CUDA-enabled environment."
if torch.cuda.device_count() > 1:
    print(f"Detected {torch.cuda.device_count()} GPUs. Multi-GPU training will be enabled.")
else:
    warnings.warn("Only one GPU detected. The script will run on a single GPU.")


os.environ["HF_TOKEN"] = ""
# The line `os.environ["CUDA_VISIBLE_DEVICES"]="0"` has been removed to allow PyTorch to see all available GPUs.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ### 2. Configuration
# We define a `Config` class to hold all our hyperparameters and settings in one place.
datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class Config:
    SEED = 42 # Use a seed for reproducibility of the random shuffle
    # The `DEVICE` attribute has been removed. `Trainer` and `device_map` will handle device placement automatically.

    MODEL_ID ="google/paligemma2-3b-pt-224"

    # Data Params
    DATASET_ID = "allenai/pixmo-points"
    NUM_SAMPLES = 100000
    IMAGE_CACHE_DIR = Path("./image_cache")
    IMAGE_CACHE_DIR.mkdir(exist_ok=True)

    # Training strategy
    USE_QLORA = True

    # QLoRA-specific configurations
    LORA_R = 8
    LORA_ALPHA = 8
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["o_proj", "k_proj", "q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

    # Training arguments

    OUTPUT_DIR = f"output/paligemma2-qlora-finetuned-multi-gpu-{datetime}"
    NUM_TRAIN_EPOCHS = 1
    # This is the batch size *per GPU*. The total effective batch size will be
    # PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 1e-4
    OPTIM = "paged_adamw_8bit"

config = Config()


# ### 3. Model Loading and Preparation
# We load the model and prepare it for training. `device_map="auto"` is key for large models,
# as it intelligently distributes the model layers across the available hardware (GPUs and CPU)
# to fit everything into memory.

processor = PaliGemmaProcessor.from_pretrained(config.MODEL_ID)

lora_config = LoraConfig(
    r=config.LORA_R,
    lora_alpha=config.LORA_ALPHA,
    lora_dropout=config.LORA_DROPOUT,
    target_modules=config.LORA_TARGET_MODULES,
    task_type="CAUSAL_LM",
    init_lora_weights="gaussian",
    use_dora=False if config.USE_QLORA else True,
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    config.MODEL_ID,
    quantization_config=bnb_config if config.USE_QLORA else None,
    device_map="auto", # Automatically distributes the model across available GPUs
    torch_dtype=torch.bfloat16
)
model.use_cache = False
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# The manual `.to(config.DEVICE)` call is removed. `device_map` handles placement.

for param in model.vision_tower.parameters():
      param.requires_grad = False
for param in model.multi_modal_projector.parameters():
      param.requires_grad = False

model.print_trainable_parameters()

# Check memory usage on the primary GPU
if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated()
    print(f"The model as is is holding: {peak_mem / 1024**3:.2f} GB of GPU RAM")

TORCH_DTYPE = model.dtype
print(f"Model dtype: {TORCH_DTYPE}")


# ### 4. Data Preparation (Unchanged)
# This section remains the same. It handles downloading, caching, and filtering the image data.

# [Data Preparation code from the original script remains here - it is unchanged]
# Load the dataset from the Hugging Face Hub
ds = load_dataset(config.DATASET_ID, split="train")

# Check if the dataset is larger than the number of samples we want
if len(ds) > config.NUM_SAMPLES:
    # Shuffle the dataset and select the first NUM_SAMPLES
    small_ds = ds.shuffle(seed=config.SEED).select(range(config.NUM_SAMPLES))
    print(f"Created a random subset of {len(small_ds)} samples.")
else:
    # If the dataset is smaller than our target, just use the whole thing
    small_ds = ds
    print(f"The full dataset ({len(ds)} samples) is smaller than {config.NUM_SAMPLES}, so using the full dataset.")


print("Starting to download and cache images...")
ds_with_local_images = small_ds.map(
    partial(download_and_cache_image, img_cache_dir=config.IMAGE_CACHE_DIR),
    num_proc=os.cpu_count()
)
original_rows = len(ds_with_local_images)
ds_ready = ds_with_local_images.filter(lambda x: x["download_status"] == "ok")
filtered_rows = len(ds_ready)
print("Image caching complete.")
print(f"Successfully downloaded {filtered_rows} out of {original_rows} images.")
ds_ready = ds_ready.filter(lambda x: len(x["points"]) > 0)
print(f"{len(ds_ready)}/{filtered_rows} samples after filtering")

ds_ready = ds_ready.filter(check_image, num_proc=16)
ds_ready = ds_ready.remove_columns(['image_url', 'image_sha256', 'count', 'collection_method', 'download_status'])
print(f"Final dataset size: {len(ds_ready)}")

# ### 5. Formatting Prompts and Labels (Unchanged)
# This section also remains the same.
# [Prompt and Label Formatting code from the original script remains here - it is unchanged]

instruction_list = [
        "Point to {label}\nPlease say 'This isn't in the image.' if it is not in the image.", "Point to all occurrences of \"{label}\"",
        "Point to any {label} in the image", "Point to any {label} in the image.", "Point: Where are the {label}",
        "Show me where the {label} are", "Can you show me where the {label} are?", "Show me where the {label} are",
        "Show me where a {label} is", "Show me where a {label} is.", "If there are any {label} in the image? Show me where they are.",
        "Where are the {label}?", "Generate a list of points showing where the {label} are.", "Find the \"{label}\".",
        "Find a \"{label}\".", "Locate all {label}.", "Locate an {label}.", "Locate a {label}.", "Locate every {label}.",
        "Locate {label}.", "Locate the {label}.", "Object: {label}\nInstruction: Point to the object.", "find {label}",
        "find {label}.", "Point to every {label}", "find any {label} in the picture", "Find the {label}", "Find any {label}",
        "Point to a {label}", "Point to an {label}", "Look for {label} in the image and show me where they are.",
        "Help me find an object in the image by pointing to them.\nObject: {label}.",
        "I am looking for {label}, where can they be found in the image?", "Can you see any {label} in the image? Point to them.",
        "Point out each {label} in the image.", "Point out every {label} in the image.", "Point to the {label} in the image.",
        "Locate each {label} in the image.", "Can you point out all {label} in this image?",
        "Please find {label} and show me where they are.", "If there are any {label} present, indicate their positions.",
        "If there is a {label} present, indicate its positions.", "show me all visible {label}",
]

# ### 6. Data Collator
# The `collate_fn` defines how individual samples are batched together.
# **MODIFICATION**: The `.to(config.DEVICE)` call has been removed. The `Trainer` will handle moving the batch to the correct GPU.

def collate_fn(batch):
    images = []
    prefixes = []
    suffixes = []

    for sample in batch:
        try:
            image = Image.open(sample["local_image_path"]).convert("RGB")
            image = image.resize((224, 224))
            images.append(image)
        except (IOError, FileNotFoundError) as e:
            print(f"Skipping sample due to image loading error from local cache: {e}")
            continue

        instr = random.choice(instruction_list)
        label = "<image>" + instr.format(label=sample['label'])
        prefixes.append(label)

        points = np.array([[p["x"], p["y"]] for p in sample["points"]])
        points_str = points_to_text(points.tolist(), sample['label'], sample['label'])
        suffixes.append(points_str)

    # --- CHANGE ---
    # The .to(config.DEVICE) call is removed from this processor call.
    # The Trainer will automatically place the data on the correct GPU.
    inputs = processor(
        text=prefixes,
        images=images,
        suffix=suffixes,
        padding="longest",
        return_tensors="pt",
    ).to(TORCH_DTYPE)
    return inputs

# ### 7. Training
# We configure `TrainingArguments` and initialize the `Trainer`. The `Trainer` will automatically
# detect and use all available GPUs for data-parallel training. It wraps the model in
# `torch.nn.DataParallel` behind the scenes.
#
# **For even better performance on multi-GPU setups, consider using DistributedDataParallel (DDP).**
# You can enable this by launching the script with `torchrun`:
# `torchrun --nproc_per_node=NUM_GPUS your_script_name.py`
# The `Trainer` will automatically detect this and switch from `DataParallel` to `DDP`.

args = TrainingArguments(
    num_train_epochs=config.NUM_TRAIN_EPOCHS,
    remove_unused_columns=False,
    per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=100,
    optim=config.OPTIM,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    output_dir=config.OUTPUT_DIR,
    eval_steps=500,
    bf16=True, # bf16 is highly recommended for Ampere and newer GPUs
    # The following line enables DDP if the script is launched with torchrun,
    # otherwise it falls back to DataParallel or single-GPU.
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
    report_to=["tensorboard"],
    dataloader_pin_memory=False
)

train_val_split = ds_ready.train_test_split(test_size=0.1, seed=42)
train_ds = train_val_split['train']
val_ds = train_val_split['test']

trainer = Trainer(
    model=model,
    train_dataset=train_ds, # Correctly use the training split
    eval_dataset=val_ds,   # Correctly use the validation split
    data_collator=collate_fn,
    args=args
)

trainer.train()


# ### 8. Save Final Model
print(f"Saving fine-tuned model to {config.OUTPUT_DIR}")
trainer.save_model(config.OUTPUT_DIR)
processor.save_pretrained(config.OUTPUT_DIR)
