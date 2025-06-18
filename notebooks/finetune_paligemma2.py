#!/usr/bin/env python
# coding: utf-8

# # Fine-Tuning PaliGemma-2 for Visual Pointing Tasks
# 
# This notebook walks through the process of fine-tuning the google/paligemma2-3b-pt-224 model to identify objects in an image and specify their coordinates.
# 
# **Learning Objectives:**
# - Load and prepare a large-scale image dataset (allenai/pixmo-points).
# - Set up a powerful multimodal model (PaliGemma-2) for fine-tuning.
# - Use memory-efficient training techniques like QLoRA.
# - Define a custom data collator to format image, text, and point data.
# - Train the model using the Hugging Face Trainer.
# - Run inference and visually evaluate the model's performance.

# In[1]:


import os
import re
import random
import hashlib
import requests
from urllib.parse import urlparse, unquote, parse_qs
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from matplotlib.patches import Circle
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig, TrainingArguments, Trainer

assert torch.cuda.is_available()

os.environ["HF_TOKEN"] = "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"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ### 2. Configuration
# We define a `Config` class to hold all our hyperparameters and settings in one place. This makes the code cleaner and easier to modify.
# 
# **Key Parameters**:
# - **`MODEL_ID`**: We are using `google/paligemma2-3b-pt-224`, a powerful vision-language model.
# - **`DATASET_ID`**: The `allenai/pixmo-points` dataset contains images with associated point coordinates for various objects. [1]
# - **`USE_QLORA`**: We set this to `True` to leverage Quantized Low-Rank Adaptation (QLoRA). This technique significantly reduces memory usage by quantizing the model to 4-bit and then attaching small, trainable "LoRA" adapters. [2] This allows us to fine-tune a large 3-billion parameter model on a single GPU.
# - **`LORA_TARGET_MODULES`**: This is a crucial parameter for LoRA. It specifies which layers of the model we will attach the trainable adapters to. Targeting the attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP layers (`gate_proj`, `up_proj`, `down_proj`) is a common and effective strategy.

# In[2]:


class Config:
    SEED = 42 # Use a seed for reproducibility of the random shuffle
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MODEL_ID = "HuggingFaceTB/SmolVLM-Base"
    MODEL_ID ="google/paligemma2-3b-pt-224"

    # Data Params
    DATASET_ID = "allenai/pixmo-points"
    NUM_SAMPLES = 100000
    IMAGE_CACHE_DIR = Path("./image_cache")
    IMAGE_CACHE_DIR.mkdir(exist_ok=True)

    # Training strategy
    # Set to True to use QLoRA for memory-efficient fine-tuning
    USE_QLORA = True

    # QLoRA-specific configurations
    LORA_R = 8
    LORA_ALPHA = 8
    LORA_DROPOUT = 0.1
    # Add all linear layers of the model to the target modules
    LORA_TARGET_MODULES = ["o_proj", "k_proj", "q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

    # Training arguments
    OUTPUT_DIR = "output/paligemma2-qlora-finetuned"
    NUM_TRAIN_EPOCHS = 3
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 4 # Increases effective batch size
    LEARNING_RATE = 1e-4
    OPTIM = "paged_adamw_8bit" # Recommended for QLoRA

    # Name for the final model on the Hub
    # HUB_MODEL_ID = "SmolVLM-finetuned-pixmo-points"

config = Config()


# ### 3. Model Loading and Preparation
# Here, we load the model and prepare it for training.
# 
# 1.  **`PaliGemmaProcessor`**: This is a utility that handles both text tokenization and image preprocessing, ensuring the inputs are in the exact format the PaliGemma model expects.
# 2.  **`BitsAndBytesConfig`**: This configures the 4-bit quantization for QLoRA. `load_in_4bit=True` enables quantization, and `bnb_4bit_compute_dtype=torch.bfloat16` sets the data type for computations, which is crucial for maintaining performance on modern GPUs.
# 3.  **`LoraConfig`**: This defines the parameters for the LoRA adapters.
# 4.  **`PaliGemmaForConditionalGeneration.from_pretrained`**: We load the model, applying the quantization config directly. `device_map="auto"` intelligently distributes the model across available GPUs.
# 5.  **Freezing Parameters**: A key step in parameter-efficient fine-tuning is to freeze the original model weights. We explicitly set `requires_grad = False` for the `vision_tower` and `multi_modal_projector`. The subsequent call to `get_peft_model` will ensure that only the LoRA adapter weights are trainable.
# 6.  **`get_peft_model`**: This function from the `peft` library wraps our base model and injects the LoRA adapters according to our `lora_config`.
# 
# The output shows that we have ~10.4 million trainable parameters, which is only a tiny fraction of the original 3 billion parameters. This is the magic of QLoRA.

# In[3]:


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
    device_map="auto",
    torch_dtype=torch.bfloat16
)
# model.add_adapter(lora_config)
# model.enable_adapters()
model.use_cache = False
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model = model.to(config.DEVICE)

for param in model.vision_tower.parameters():
      param.requires_grad = False
for param in model.multi_modal_projector.parameters():
      param.requires_grad = False

print(model.get_nb_trainable_parameters())

peak_mem = torch.cuda.max_memory_allocated()
print(f"The model as is is holding: {peak_mem / 1024**3:.2f} of GPU RAM")

TORCH_DTYPE = model.dtype
print(TORCH_DTYPE)


# ### 4. Data Preparation
# The dataset contains URLs to images. We need to download them, cache them locally to speed up training, and perform some cleaning.
# 
# **Steps**:
# 1.  **Load Dataset**: Load the `train` split from the Hugging Face Hub.
# 2.  **Subset**: We take a random subset of 100,000 samples for faster processing.
# 3.  **Download and Cache**: The `download_and_cache_image` function is mapped across the dataset. It downloads each image, saves it to the `IMAGE_CACHE_DIR`, and adds the local file path to the dataset. Using `num_proc` speeds this up significantly.
# 4.  **Filter**: We filter out examples where the download failed, where there are no points, or where the image is invalid (e.g., corrupted or has an incorrect number of channels).

# In[4]:


# Load the dataset from the Hugging Face Hub
ds = load_dataset(config.DATASET_ID, split="train")


# In[5]:


# Check if the dataset is larger than the number of samples we want
if len(ds) > config.NUM_SAMPLES:
    # Shuffle the dataset and select the first NUM_SAMPLES
    small_ds = ds.shuffle(seed=config.SEED).select(range(config.NUM_SAMPLES))
    print(f"Created a random subset of {len(small_ds)} samples.")
else:
    # If the dataset is smaller than our target, just use the whole thing
    small_ds = ds
    print(f"The full dataset ({len(ds)} samples) is smaller than {config.NUM_SAMPLES}, so using the full dataset.")


# In[6]:


def url2path(url):
    try:
        # Generate a hash of the URL for uniqueness
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        # Parse URL and get base filename
        parsed_url = urlparse(url)
        base_filename = os.path.basename(parsed_url.path)
        base_filename = unquote(base_filename)
        
        # Remove query parameters and clean filename
        base_filename = base_filename.split('?')[0]
        base_filename = re.sub(r'[^A-Za-z0-9._-]', '_', base_filename)
        
        # Get extension from URL or default to jpg
        ext = os.path.splitext(base_filename)[1].lower()
        if not ext or ext not in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
            # Check query parameters for format
            query_params = parse_qs(parsed_url.query)
            if 'format' in query_params:
                format_param = query_params['format'][0].lower()
                if format_param in ['jpg', 'jpeg', 'png', 'webp', 'gif']:
                    ext = f'.{format_param}'
            else:
                ext = '.jpg'
        
        filename_length = len(base_filename)

        # Create final filename
        if not base_filename or base_filename == 'image' or filename_length > 200:
            filename = f'image_{url_hash}{ext}'
        else:
            base_filename = os.path.splitext(base_filename)[0]
            filename = f'{base_filename}_{url_hash}{ext}'

        local_path = config.IMAGE_CACHE_DIR / filename
        
        # Ensure the path is not a directory
        if local_path.is_dir():
            local_path = local_path.with_name(f'{local_path.name}_file{ext}')
            
        return local_path
        
    except Exception as e:
        print(f"Error processing URL {url}: {str(e)}")
        # Fallback to a safe filename with hash
        return config.IMAGE_CACHE_DIR / f'image_{hashlib.md5(url.encode()).hexdigest()}.jpg'


def download_and_cache_image(example):
    """
    Downloads the image from its URL, saves it to a local cache,
    and returns a dictionary with the new local path.
    """
    try:
        # Create a unique, safe filename from the image URL
        image_url = example["image_url"]
        # Use the last part of the URL as a filename, which is usually unique
        local_path = url2path(image_url)

        # Only download if the file doesn't already exist in the cache
        if not local_path.exists():
            response = requests.get(image_url, timeout=10, stream=True)
            response.raise_for_status()
            
            # Save the image to the local cache
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Return the new column
        return {"local_image_path": str(local_path), "download_status": "ok"}
    
    except Exception as e:
        # If download fails, mark it so we can filter it out later
        return {"local_image_path": None, "download_status": "error"}

# Use `map` to apply the download function to the entire dataset.
# We set `num_proc` to a higher value to parallelize the download process.
# This will show a progress bar.
print("Starting to download and cache images...")
ds_with_local_images = small_ds.map(
    download_and_cache_image,
    num_proc=os.cpu_count()  # Use half of the available CPU cores
)

# Filter out any examples that failed to download
original_rows = len(ds_with_local_images)
ds_ready = ds_with_local_images.filter(lambda x: x["download_status"] == "ok")
filtered_rows = len(ds_ready)

print("Image caching complete.")
print(f"Successfully downloaded {filtered_rows} out of {original_rows} images.")

# filter samples that have no points
ds_ready = ds_ready.filter(lambda x: len(x["points"]) > 0)
print(f"{len(ds_ready)}/{filtered_rows} samples after filtering")


def check_image(x):
    try:
        img = Image.open(x["local_image_path"]).convert("RGB")
        img = np.array(img)
        if img.shape[2] != 3 or img.shape[1] == 1:
            return False
        return True
    except:
        return False

# This runs the function in parallel across 4 separate processes
# It's often faster for CPU-bound or local disk I/O tasks.
ds_ready = ds_ready.filter(check_image, num_proc=16)
ds_ready = ds_ready.remove_columns(['image_url', 'image_sha256', 'count', 'collection_method', 'download_status'])


# In[7]:


len(ds_ready)


# ### 5. Formatting Prompts and Labels
# The model needs to be trained in a chat-like format. We define:
# - **`instruction_list`**: A list of various ways to ask the model to point to an object. During training, we randomly pick one to make the model robust to different phrasings.
# - **`points_to_text`**: This function converts a list of `(x, y)` coordinates into the target string format (`<point ...>` or `<points ...>`). This will be our label.
# - **`text_to_points`**: The inverse function, useful for parsing the model's output during inference and evaluation.

# In[8]:


instruction_list = [
        "Point to {label}\nPlease say 'This isn't in the image.' if it is not in the image.",
        "Point to all occurrences of \"{label}\"",
        "Point to any {label} in the image",
        "Point to any {label} in the image.",
        "Point: Where are the {label}",
        "Show me where the {label} are",
        "Can you show me where the {label} are?",
        "Show me where the {label} are",
        "Show me where a {label} is",
        "Show me where a {label} is.",
        "If there are any {label} in the image? Show me where they are.",
        "Where are the {label}?",
        "Generate a list of points showing where the {label} are.",
        "Find the \"{label}\".",
        "Find a \"{label}\".",
        "Locate all {label}.",
        "Locate an {label}.",
        "Locate a {label}.",
        "Locate every {label}.",
        "Locate {label}.",
        "Locate the {label}.",
        "Object: {label}\nInstruction: Point to the object.",
        "find {label}",
        "find {label}.",
        "Point to every {label}",
        "find any {label} in the picture",
        "Find the {label}",
        "Find any {label}",
        "Point to a {label}",
        "Point to an {label}",
        "Look for {label} in the image and show me where they are.",
        "Help me find an object in the image by pointing to them.\nObject: {label}.",
        "I am looking for {label}, where can they be found in the image?",
        "Can you see any {label} in the image? Point to them.",
        "Point out each {label} in the image.",
        "Point out every {label} in the image.",
        "Point to the {label} in the image.",
        "Locate each {label} in the image.",
        "Can you point out all {label} in this image?",
        "Please find {label} and show me where they are.",
        "If there are any {label} present, indicate their positions.",
        "If there is a {label} present, indicate its positions.",
        "show me all visible {label}",
]


def points_to_text(points, label_text, alt_text):
    if len(points) == 1:
        x_str, y_str = points[0]
        return f"<point x=\"{x_str}\" y=\"{y_str}\" alt=\"{alt_text}\">{label_text}</point>"
    point_text = []
    for ix, (x, y) in enumerate(points, start=1):
        point_text.append(f"x{ix}=\"{x}\"")
        point_text.append(f"y{ix}=\"{y}\"")
    point_text = " ".join(point_text)
    return f"<points {point_text} alt=\"{alt_text}\">{label_text}</points>"


def text_to_points(text):
    # Single point pattern with flexible spaces and non-greedy label match
    single_point_pattern = r'^<point\s+x="([^"]+)"\s+y="([^"]+)"\s+alt="([^"]+)">(.*?)</point>$'
    m = re.match(single_point_pattern, text, re.DOTALL)
    if m:
        x_str, y_str, alt_text, label_text = m.groups()
        points = [(x_str, y_str)]
        return points, label_text, alt_text

    # Multiple points pattern
    if text.startswith("<points"):
        alt_match = re.search(r'alt="([^"]+)"', text)
        label_match = re.search(r'>(.*?)</points>', text, re.DOTALL)
        if not alt_match or not label_match:
            raise ValueError("Invalid format for multiple points")
        alt_text = alt_match.group(1)
        label_text = label_match.group(1)

        x_matches = re.findall(r'x(\d+)="([^"]+)"', text)
        y_matches = re.findall(r'y(\d+)="([^"]+)"', text)

        x_dict = {int(idx): val for idx, val in x_matches}
        y_dict = {int(idx): val for idx, val in y_matches}

        points = []
        for i in sorted(x_dict.keys()):
            if i in y_dict:
                points.append((x_dict[i], y_dict[i]))
            else:
                raise ValueError(f"Missing y{i} value")

        return points, label_text, alt_text

    raise ValueError("Input string does not start with <point> or <points>")


# ### 6. Data Collator
# The `collate_fn` is a critical piece. The `Trainer` takes individual samples from the dataset and groups them into a batch. This function defines *how* that happens.
# 
# For each sample in the batch, it:
# 1.  Opens the cached image and resizes it to 224x224, the expected input size for the model's vision tower.
# 2.  Randomly selects an instruction from our `instruction_list` and formats it with the sample's label.
# 3.  Constructs the target output string using `points_to_text`.
# 4.  Finally, it uses the `PaliGemmaProcessor` to convert the batch of images, prefixes (instructions), and suffixes (target point strings) into a dictionary of tensors (`input_ids`, `attention_mask`, `pixel_values`, `labels`) ready to be fed into the model.

# In[9]:


def debug_vis_img_point(img, point_str):
    img = np.array(img)
    plt.imshow(img)
    points, label, alt_text = text_to_points(point_str)
    for p in points:
        x, y = list(map(float, p))
        h, w = img.shape[:2]
        circle = Circle(((x / 100) * h, (y / 100) * w), 5, fill=False, edgecolor='red', linewidth=2)
        ax = plt.gca()
        ax.add_patch(circle)
    ax.set_title(f"{label}")
    plt.axis('equal')
    plt.show()
    print(point_str)


def collate_fn(batch):
    images = []
    prefixes = []  # what will be sent along with image
    suffixes = []  # what's expected

    for sample in batch:
        # prepare inputs
        try:
            image = Image.open(sample["local_image_path"]).convert("RGB")

            # make resize here
            image = image.resize((224, 224))
            images.append(image)
        except (IOError, FileNotFoundError) as e:
            print(f"Skipping sample due to image loading error from local cache: {e}")
            continue

        instr = random.choice(instruction_list)
        label = "<image>" + instr.format(label=sample['label'])
        prefixes.append(label)

        # prepare what's expected
        w, h = image.size
        points = np.stack([[p["x"], p["y"]] for p in sample["points"]])
        points_str = points_to_text(points, sample['label'], sample['label'])
        suffixes.append(points_str)

    # For Debug Uncomment when needed
    # for i in range(2):
    #     debug_vis_img_point(images[i], suffixes[i])

    inputs = processor(
        text=prefixes,
        images=images,
        suffix=suffixes,
        padding="longest",
        return_tensors="pt",
    ).to(TORCH_DTYPE).to(config.DEVICE)
    return inputs


sample_batch = collate_fn(ds_ready.select(range(2)))
print("sample_batch keys:")
for k, v in sample_batch.items():
    print("\t", k, v.shape)

print(sample_batch["input_ids"])


# ### 7. Training
# 
# **Mistake Corrected**: The original code created a train/validation split (`train_val_split`) but then passed the entire `ds_ready` to the `Trainer`'s `train_dataset` argument. This means the model would not be evaluated on a hold-out set during training. I've corrected this by passing `train_ds` to `train_dataset` and `val_ds` to `eval_dataset`.
# 
# 1.  **`TrainingArguments`**: We configure the training process, setting the number of epochs, batch size, learning rate, and other parameters from our `Config` class. We enable `bf16=True` for mixed-precision training, which speeds up computation and reduces memory usage. `gradient_checkpointing=True` is another memory-saving technique.
# 2.  **Train/Validation Split**: We split our prepared dataset into a 90% training set and a 10% validation set.
# 3.  **`Trainer`**: We initialize the `Trainer` with our model, training arguments, datasets, and the custom `collate_fn`.
# 4.  **`trainer.train()`**: This command starts the fine-tuning process. The `KeyboardInterrupt` in your original notebook indicates that this step was stopped manually. **You will need to run this cell and let it complete fully.** This will take a significant amount of time.

# In[10]:


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
    gradient_checkpointing=True,
    bf16=True,
    # report_to=["tensorboard"],
    dataloader_pin_memory=False
)

train_val_split = ds_ready.train_test_split(test_size=0.1, seed=42)
train_ds = train_val_split['train']
val_ds = train_val_split['test']

trainer = Trainer(
    model=model,
    train_dataset=ds_ready,
    data_collator=collate_fn,
    args=args
)


# In[11]:


trainer.train()


# ### 8. Save Final Model
# After training is complete, it's crucial to save your work. We save the trained model adapters and the processor. This allows us to easily load the fine-tuned model later for inference without having to repeat the training.

# In[ ]:


print(f"Saving fine-tuned model to {config.OUTPUT_DIR}")
trainer.save_model(config.OUTPUT_DIR)
processor.save_pretrained(config.OUTPUT_DIR)


# ### 9. Inference and Evaluation
# 
# Now for the exciting part: let's see how well our model performs! This section was missing from the original notebook.
# 
# **Inference Steps**:
# 1.  **Load Model**: We load the base model again, but this time we also load the fine-tuned LoRA adapters from our output directory using `PeftModel.from_pretrained`.
# 2.  **Prepare Inputs**: We'll take a sample from our validation set (`val_ds`). We format a prompt just like we did in training.
# 3.  **Generate**: We use `model.generate()` to get the model's prediction. `max_new_tokens` controls the maximum length of the output.
# 4.  **Decode and Visualize**: We decode the generated token IDs back into a string. We then use a helper function, `run_and_visualize`, to parse the predicted points and draw them on the original image, which provides an intuitive way to assess the model's accuracy.

# In[ ]:


from peft import PeftModel

# Load the base model and processor from the saved directory
print(f"Loading model from {config.OUTPUT_DIR}")
base_model = PaliGemmaForConditionalGeneration.from_pretrained(
    config.MODEL_ID,
    quantization_config=bnb_config, # Must use the same quantization config
    device_map="auto",
    torch_dtype=torch.bfloat16
)
processor = PaliGemmaProcessor.from_pretrained(config.OUTPUT_DIR)
model = PeftModel.from_pretrained(base_model, config.OUTPUT_DIR)
model = model.to(config.DEVICE)
model.eval() # Set the model to evaluation mode
print("Model loaded successfully.")


# In[ ]:


from matplotlib.patches import Circle

def run_and_visualize(sample):
    """Runs inference on a single sample and visualizes the result."""
    image = Image.open(sample["local_image_path"]).convert("RGB")
    label_text = sample['label']
    
    # Prepare the prompt
    prompt = f"<image>\nPoint to {label_text}"
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(config.DEVICE)
    
    # Generate output
    with torch.no_grad():
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    
    # Decode and parse
    generated_text = processor.decode(generation[0], skip_special_tokens=True)
    # We need to extract just the generated part (the <point> string)
    prediction_str = generated_text.split("\n")[-1]
    predicted_points, _, _ = text_to_points(prediction_str)

    # Get ground truth points
    gt_points = [(p['x'], p['y']) for p in sample["points"]]
    
    # Visualize
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    ax = plt.gca()
    ax.set_title(f"Prompt: 'Point to {label_text}'")

    # Draw ground truth points (blue circles)
    for x, y in gt_points:
        h, w = image.height, image.width
        circle = Circle((x * w, y * h), radius=8, fill=False, edgecolor='blue', linewidth=2, linestyle='--')
        ax.add_patch(circle)

    # Draw predicted points (red 'X')
    for x, y in predicted_points:
        h, w = image.height, image.width
        ax.scatter(x * w, y * h, marker='x', color='red', s=100, linewidths=3)

    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='blue', marker='o', linestyle='--', label='Ground Truth', markersize=10, markerfacecolor='none'),
                       Line2D([0], [0], marker='x', color='red', label='Prediction', markersize=10, linestyle='None')]
    ax.legend(handles=legend_elements, loc='best')
    plt.axis('off')
    plt.show()
    
    print("Ground Truth:", points_to_text(gt_points, label_text, label_text))
    print("Prediction:", prediction_str)

# Run visualization on a few random samples from the validation set
for i in range(5):
    print(f"--- Sample {i+1} ---")
    random_index = random.randint(0, len(val_ds) - 1)
    run_and_visualize(val_ds[random_index])


# ## Quantitative Evaluation Metric:
# 
# Visual inspection is good, but a numerical score is better for rigorous evaluation.
# 
# **Object-based Pointing Accuracy (OPA)**. For a given prediction, if the distance to the nearest ground truth point is within a certain threshold (e.g., 15% of the image diagonal), it's considered a true positive. You could then calculate Precision, Recall, and F1-score over the entire validation set.

# In[ ]:


import numpy as np
import torch
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

def calculate_opa_metrics(predicted_points, gt_points, threshold=0.05):
    """
    Calculates Object-based Pointing Accuracy (OPA) metrics (Precision, Recall, F1).

    A predicted point is considered a True Positive (TP) if its normalized distance
    to the NEAREST ground truth point is within a given threshold.

    Args:
        predicted_points (list of tuples): List of (x, y) coordinates for predicted points.
        gt_points (list of tuples): List of (x, y) coordinates for ground truth points.
        threshold (float): The maximum normalized distance for a point to be considered a match.
                           The distance is normalized by the image diagonal (sqrt(2)),
                           so 0.05 corresponds to ~5% of the diagonal.

    Returns:
        dict: A dictionary containing precision, recall, and f1_score.
    """
    # Handle edge case where there are no predictions.
    if not predicted_points:
        # If there are also no ground truth points, it's a perfect match.
        if not gt_points:
            return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0}
        # If there are ground truth points, we missed all of them.
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

    # Handle edge case where there are no ground truth points, but we predicted some.
    # This is a pure hallucination scenario.
    if not gt_points:
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

    predicted_array = np.array(predicted_points)
    gt_array = np.array(gt_points)

    # Calculate the pairwise distances between all predicted and ground truth points.
    # The result is a matrix of shape (num_predicted, num_gt).
    distances = cdist(predicted_array, gt_array)

    # For each predicted point, find the distance to the nearest ground truth point.
    min_distances = np.min(distances, axis=1)

    # A prediction is a "hit" (True Positive) if its nearest GT point is within the threshold.
    true_positives = np.sum(min_distances <= threshold)

    # Calculate Precision and Recall
    precision = true_positives / len(predicted_points)
    recall = true_positives / len(gt_points)

    # Calculate F1 Score
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}


# --- Main Evaluation Loop ---
print("Starting quantitative evaluation over the entire validation set...")

# Ensure the model is in evaluation mode
model.eval()

all_precisions = []
all_recalls = []
all_f1s = []

# Use torch.no_grad() for the entire loop to save memory and speed up inference
with torch.no_grad():
    # Iterate over the validation dataset with a progress bar
    for sample in tqdm(val_ds, desc="Evaluating"):
        image = Image.open(sample["local_image_path"]).convert("RGB")
        label_text = sample['label']

        # 1. Prepare the prompt for the model
        prompt = f"<image>\nPoint to {label_text}"
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(config.DEVICE)

        # 2. Generate the model's output
        generation = model.generate(**inputs, max_new_tokens=150, do_sample=False)
        generated_text = processor.decode(generation[0], skip_special_tokens=True)

        # 3. Parse the output to get predicted points
        # The model's full output includes the prompt, so we extract the last part.
        prediction_str = generated_text.split("\n")[-1].strip()
        predicted_points, _, _ = text_to_points(prediction_str)

        # 4. Get ground truth points
        gt_points = [(p['x'], p['y']) for p in sample["points"]]

        # 5. Calculate metrics for this sample and store them
        metrics = calculate_opa_metrics(predicted_points, gt_points, threshold=0.05)
        all_precisions.append(metrics['precision'])
        all_recalls.append(metrics['recall'])
        all_f1s.append(metrics['f1_score'])

# --- Report Final Scores ---
mean_precision = np.mean(all_precisions)
mean_recall = np.mean(all_recalls)
mean_f1 = np.mean(all_f1s)

print("\n--- Quantitative Evaluation Results ---")
print(f"Validation Set Size: {len(val_ds)}")
print(f"Average Precision: {mean_precision:.4f}")
print(f"Average Recall:    {mean_recall:.4f}")
print(f"Average F1-Score:  {mean_f1:.4f}")
print("---------------------------------------")

