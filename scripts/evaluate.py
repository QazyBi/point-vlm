from pathlib import Path
import json
import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

import torch
from PIL import Image
from matplotlib import pyplot as plt
from peft import LoraConfig
from datasets import load_dataset
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from utils import points_to_text, text_to_points, download_and_cache_image, check_image
from matplotlib.patches import Circle
from functools import partial
import os
import random

# os.environ["HF_TOKEN"] = "token"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class Config:
    SEED = 42 # Use a seed for reproducibility of the random shuffle

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

    OUTPUT_DIR = f"output/paligemma2-qlora-finetuned-multi-gpu-2025-06-18_16-56-27/checkpoint-12000"
    NUM_TRAIN_EPOCHS = 1
    # This is the batch size *per GPU*. The total effective batch size will be
    # PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 1e-4
    OPTIM = "paged_adamw_8bit"

    # Evaluation artifacts configuration
    ARTIFACTS_DIR = Path("./evaluation_artifacts")
    EVAL_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    EVAL_DIR = ARTIFACTS_DIR / f"eval_{EVAL_TIMESTAMP}"
    IMAGES_DIR = EVAL_DIR / "visualizations"
    METRICS_DIR = EVAL_DIR / "metrics"
    LOGS_DIR = EVAL_DIR / "logs"
    REPORTS_DIR = EVAL_DIR / "reports"

config = Config()

# Create evaluation artifacts directory structure
config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize metrics storage
evaluation_results = {
    "model_info": {
        "model_id": config.MODEL_ID,
        "output_dir": config.OUTPUT_DIR,
        "evaluation_timestamp": config.EVAL_TIMESTAMP
    },
    "dataset_info": {
        "dataset_id": config.DATASET_ID,
        "num_samples": config.NUM_SAMPLES
    },
    "predictions": [],
    "metrics": {}
}

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


# ### 9. Inference and Evaluation
# **MODIFICATION**: For inference, we load the model with `device_map="auto"` which will again spread it across GPUs.
# Input tensors are moved to `model.device` to ensure they are on the same primary device as the model's starting layers.

from peft import PeftModel

# Load the base model and processor from the saved directory
print(f"Loading model from {config.OUTPUT_DIR}")
base_model = PaliGemmaForConditionalGeneration.from_pretrained(
    config.MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto", # Use auto device map for efficient inference
    torch_dtype=torch.bfloat16
)
processor = PaliGemmaProcessor.from_pretrained(config.MODEL_ID)
# The model is loaded onto the device(s) by PeftModel using the base_model's device_map
model = PeftModel.from_pretrained(base_model, config.OUTPUT_DIR)
model.eval() # Set the model to evaluation mode
print("Model loaded successfully.")

def calculate_point_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    print(point1, point2)
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_metrics(gt_points, pred_points, image_width, image_height):
    """Calculate various metrics for point prediction evaluation."""
    if not gt_points or not pred_points:
        return {
            "distance_error": float('inf'),
            "normalized_distance_error": float('inf'),
            "mae_x": float('inf'),
            "mae_y": float('inf'),
            "mse_x": float('inf'),
            "mse_y": float('inf'),
            "success_rate": 0.0
        }
    
    # Calculate distances between ground truth and predicted points
    distances = []
    mae_x_errors = []
    mae_y_errors = []
    mse_x_errors = []
    mse_y_errors = []
    
    for gt_point in gt_points:
        min_distance = float('inf')
        min_mae_x = float('inf')
        min_mae_y = float('inf')
        min_mse_x = float('inf')
        min_mse_y = float('inf')
        
        for pred_point in pred_points:
            distance = calculate_point_distance(gt_point, pred_point)
            if distance < min_distance:
                min_distance = distance
                min_mae_x = abs(gt_point[0] - pred_point[0])
                min_mae_y = abs(gt_point[1] - pred_point[1])
                min_mse_x = (gt_point[0] - pred_point[0])**2
                min_mse_y = (gt_point[1] - pred_point[1])**2
        
        distances.append(min_distance)
        mae_x_errors.append(min_mae_x)
        mae_y_errors.append(min_mae_y)
        mse_x_errors.append(min_mse_x)
        mse_y_errors.append(min_mse_y)
    
    # Calculate success rate (points within a reasonable threshold)
    threshold = 0.05  # 5% of image diagonal
    diagonal = np.sqrt(image_width**2 + image_height**2)
    success_threshold = threshold * diagonal
    success_count = sum(1 for d in distances if d <= success_threshold)
    success_rate = success_count / len(distances) if distances else 0.0
    
    return {
        "distance_error": np.mean(distances),
        "normalized_distance_error": np.mean(distances) / diagonal,
        "mae_x": np.mean(mae_x_errors),
        "mae_y": np.mean(mae_y_errors),
        "mse_x": np.mean(mse_x_errors),
        "mse_y": np.mean(mse_y_errors),
        "success_rate": success_rate,
        "num_gt_points": len(gt_points),
        "num_pred_points": len(pred_points)
    }

def run_and_visualize(sample, sample_idx=None, save_image=True):
    """Runs inference on a single sample and visualizes the result."""
    image = Image.open(sample["local_image_path"]).convert("RGB")
    label_text = sample['label']

    prompt = f"<image>\nPoint to {label_text}"
    # --- CHANGE ---
    # Move inputs to model.device, which is the primary device assigned by device_map
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    generated_text = processor.decode(generation[0], skip_special_tokens=True)
    try:
        prediction_str = generated_text.split("\n")[-1]
        predicted_points, _, _ = text_to_points(prediction_str)
        predicted_points = [(float(x), float(y)) for x, y in predicted_points]
    except (ValueError, IndexError):
        logger.warning(f"Could not parse model output: {prediction_str}")
        predicted_points = []

    gt_points = [(p['x'], p['y']) for p in sample["points"]]

    # Calculate metrics
    metrics = calculate_metrics(gt_points, predicted_points, image.width, image.height)
    
    # Store prediction results
    prediction_result = {
        "sample_idx": sample_idx,
        "label": label_text,
        "gt_points": gt_points,
        "predicted_points": predicted_points,
        "prediction_text": prediction_str if 'prediction_str' in locals() else "Parsing Failed",
        "metrics": metrics
    }
    evaluation_results["predictions"].append(prediction_result)

    if save_image:
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        ax = plt.gca()
        ax.set_title(f"Prompt: 'Point to {label_text}'")

        h, w = image.height, image.width
        for x, y in gt_points:
            circle = Circle(((float(x) / 100) * w, (float(y) / 100) * h), radius=8, fill=False, edgecolor='blue', linewidth=2, linestyle='--')
            ax.add_patch(circle)

        for x, y in predicted_points:
            ax.scatter((float(x) / 100) * w, (float(y) / 100) * h, marker='x', color='red', s=100, linewidths=3)

        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='blue', marker='o', linestyle='--', label='Ground Truth', markersize=10, markerfacecolor='none'),
                           Line2D([0], [0], marker='x', color='red', label='Prediction', markersize=10, linestyle='None')]
        ax.legend(handles=legend_elements, loc='best')
        plt.axis('off')
        
        # Save the visualization
        if sample_idx is not None:
            image_filename = f"sample_{sample_idx:04d}_{label_text.replace(' ', '_')}.png"
        else:
            image_filename = f"sample_{label_text.replace(' ', '_')}_{random.randint(1000, 9999)}.png"
        
        image_path = config.IMAGES_DIR / image_filename
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization: {image_path}")
        logger.info(f"Ground Truth: {points_to_text(gt_points, label_text, label_text)}")
        logger.info(f"Prediction: {prediction_str if 'prediction_str' in locals() else 'Parsing Failed'}")
        logger.info(f"Metrics: {metrics}")

    return prediction_result

def calculate_overall_metrics():
    """Calculate overall metrics from all predictions."""
    if not evaluation_results["predictions"]:
        return {}
    
    all_metrics = [pred["metrics"] for pred in evaluation_results["predictions"]]
    
    # Filter out infinite values
    valid_metrics = [m for m in all_metrics if m["distance_error"] != float('inf')]
    
    if not valid_metrics:
        return {"error": "No valid predictions found"}
    
    overall_metrics = {
        "num_samples": len(evaluation_results["predictions"]),
        "num_valid_predictions": len(valid_metrics),
        "mean_distance_error": np.mean([m["distance_error"] for m in valid_metrics]),
        "mean_normalized_distance_error": np.mean([m["normalized_distance_error"] for m in valid_metrics]),
        "mean_mae_x": np.mean([m["mae_x"] for m in valid_metrics]),
        "mean_mae_y": np.mean([m["mae_y"] for m in valid_metrics]),
        "mean_mse_x": np.mean([m["mse_x"] for m in valid_metrics]),
        "mean_mse_y": np.mean([m["mse_y"] for m in valid_metrics]),
        "mean_success_rate": np.mean([m["success_rate"] for m in valid_metrics]),
        "std_distance_error": np.std([m["distance_error"] for m in valid_metrics]),
        "median_distance_error": np.median([m["distance_error"] for m in valid_metrics])
    }
    
    return overall_metrics

def save_evaluation_artifacts():
    """Save all evaluation artifacts to files."""
    # Calculate overall metrics
    overall_metrics = calculate_overall_metrics()
    evaluation_results["metrics"] = overall_metrics
    
    # Save detailed results as JSON
    results_file = config.METRICS_DIR / "detailed_results.json"
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    logger.info(f"Saved detailed results: {results_file}")
    
    # Save metrics summary as CSV
    if evaluation_results["predictions"]:
        metrics_df = pd.DataFrame([pred["metrics"] for pred in evaluation_results["predictions"]])
        metrics_file = config.METRICS_DIR / "metrics_summary.csv"
        metrics_df.to_csv(metrics_file, index=False)
        logger.info(f"Saved metrics summary: {metrics_file}")
    
    # Save overall metrics as JSON
    overall_metrics_file = config.METRICS_DIR / "overall_metrics.json"
    with open(overall_metrics_file, 'w') as f:
        json.dump(overall_metrics, f, indent=2, default=str)
    logger.info(f"Saved overall metrics: {overall_metrics_file}")
    
    # Create evaluation report
    create_evaluation_report(overall_metrics)

def create_evaluation_report(overall_metrics):
    """Create a comprehensive evaluation report."""
    report_file = config.REPORTS_DIR / "evaluation_report.md"
    
    with open(report_file, 'w') as f:
        f.write("# Point-VLM Evaluation Report\n\n")
        f.write(f"**Evaluation Date:** {config.EVAL_TIMESTAMP}\n\n")
        
        f.write("## Model Information\n")
        f.write(f"- **Model ID:** {config.MODEL_ID}\n")
        f.write(f"- **Output Directory:** {config.OUTPUT_DIR}\n")
        f.write(f"- **Dataset:** {config.DATASET_ID}\n")
        f.write(f"- **Number of Samples:** {config.NUM_SAMPLES}\n\n")
        
        f.write("## Overall Metrics\n")
        if "error" not in overall_metrics:
            f.write(f"- **Total Samples:** {overall_metrics.get('num_samples', 'N/A')}\n")
            f.write(f"- **Valid Predictions:** {overall_metrics.get('num_valid_predictions', 'N/A')}\n")
            f.write(f"- **Mean Distance Error:** {overall_metrics.get('mean_distance_error', 'N/A'):.4f}\n")
            f.write(f"- **Mean Normalized Distance Error:** {overall_metrics.get('mean_normalized_distance_error', 'N/A'):.4f}\n")
            f.write(f"- **Mean MAE X:** {overall_metrics.get('mean_mae_x', 'N/A'):.4f}\n")
            f.write(f"- **Mean MAE Y:** {overall_metrics.get('mean_mae_y', 'N/A'):.4f}\n")
            f.write(f"- **Mean MSE X:** {overall_metrics.get('mean_mse_x', 'N/A'):.4f}\n")
            f.write(f"- **Mean MSE Y:** {overall_metrics.get('mean_mse_y', 'N/A'):.4f}\n")
            f.write(f"- **Mean Success Rate:** {overall_metrics.get('mean_success_rate', 'N/A'):.4f}\n")
            f.write(f"- **Std Distance Error:** {overall_metrics.get('std_distance_error', 'N/A'):.4f}\n")
            f.write(f"- **Median Distance Error:** {overall_metrics.get('median_distance_error', 'N/A'):.4f}\n")
        else:
            f.write(f"- **Error:** {overall_metrics['error']}\n")
        
        f.write("\n## Artifacts\n")
        f.write(f"- **Visualizations:** {config.IMAGES_DIR}\n")
        f.write(f"- **Metrics:** {config.METRICS_DIR}\n")
        f.write(f"- **Logs:** {config.LOGS_DIR}\n")
        f.write(f"- **Reports:** {config.REPORTS_DIR}\n")
    
    logger.info(f"Created evaluation report: {report_file}")


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

train_val_split = ds_ready.train_test_split(test_size=0.1, seed=42)
train_ds = train_val_split['train']
val_ds = train_val_split['test']

logger.info("Starting comprehensive evaluation...")

# Run evaluation on validation set
logger.info(f"Running evaluation on {len(val_ds)} validation samples...")

# Run evaluation on a subset for visualization (first 10 samples)
visualization_samples = min(10, len(val_ds))
logger.info(f"Running visualization on {visualization_samples} samples...")

for i in range(visualization_samples):
    logger.info(f"Processing sample {i+1}/{visualization_samples}")
    run_and_visualize(val_ds[i], sample_idx=i, save_image=True)

# Run evaluation on a larger subset for metrics (first 100 samples)
metrics_samples = min(100, len(val_ds))
logger.info(f"Running metrics evaluation on {metrics_samples} samples...")

for i in range(visualization_samples, metrics_samples):
    if i % 10 == 0:
        logger.info(f"Processing sample {i+1}/{metrics_samples}")
    run_and_visualize(val_ds[i], sample_idx=i, save_image=False)

# Save all evaluation artifacts
logger.info("Saving evaluation artifacts...")
save_evaluation_artifacts()

logger.info(f"Evaluation complete! Results saved to: {config.EVAL_DIR}")

# Print summary
overall_metrics = evaluation_results["metrics"]
if "error" not in overall_metrics:
    logger.info("=== EVALUATION SUMMARY ===")
    logger.info(f"Total samples evaluated: {overall_metrics.get('num_samples', 'N/A')}")
    logger.info(f"Valid predictions: {overall_metrics.get('num_valid_predictions', 'N/A')}")
    logger.info(f"Mean distance error: {overall_metrics.get('mean_distance_error', 'N/A'):.4f}")
    logger.info(f"Mean success rate: {overall_metrics.get('mean_success_rate', 'N/A'):.4f}")
else:
    logger.error(f"Evaluation failed: {overall_metrics['error']}")

print(f"\n--- Evaluation artifacts saved to: {config.EVAL_DIR} ---")
print(f"--- Check the evaluation report at: {config.REPORTS_DIR / 'evaluation_report.md'} ---")
