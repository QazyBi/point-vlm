#!/usr/bin/env python3
"""
Script to run Point-VLM evaluation with artifact saving.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path to import utils
sys.path.append(str(Path(__file__).parent.parent))

def main():
    """Run the evaluation script."""
    try:
        # Import and run the evaluation
        from evaluate import logger, config
        
        logger.info("Starting Point-VLM evaluation...")
        logger.info(f"Model: {config.MODEL_ID}")
        logger.info(f"Dataset: {config.DATASET_ID}")
        logger.info(f"Output directory: {config.EVAL_DIR}")
        
        # The evaluation will run automatically when the module is imported
        # since the main execution code is at the bottom of evaluate.py
        
        logger.info("Evaluation completed successfully!")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r scripts/requirements_eval.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
