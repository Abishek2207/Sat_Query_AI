"""
Remote Sensing AI System - Main Entry Point.

Unified Command Line Interface for:
- Creating sample satellite datasets (RGB, GeoTIFF, SAR)
- Model Training (with GPU/CPU auto-detection and early stopping)
- Model Evaluation (with Confusion Matrix and IoU metrics)
- Satellite Image Inference & Feature Extraction (with multi-panel plots)
- End-to-End Quick Demo
"""

import os
import sys
import argparse
from typing import Optional

from utils.sample_generator import generate_synthetic_dataset
from training.train import train_model, load_config
from evaluation.evaluate import evaluate_model_checkpoint
from inference.predict import predict_image, print_prediction_cli


def run_create_samples(output_dir: str = "data/raw/EuroSAT_Sample", count: int = 12) -> None:
    """Generates synthetic multi-modal satellite imagery."""
    print(f"\n[Sample Generator] Generating {count} synthetic satellite images per class...")
    path = generate_synthetic_dataset(
        output_dir=output_dir,
        num_samples_per_class=count,
        size=(64, 64),
        include_geotiff=True
    )
    print(f"[Sample Generator] Complete! Dataset saved to: {os.path.abspath(path)}")
    print(f"[Sample Generator] Standalone test samples created in: 'data/raw/samples/'")


def run_train(args) -> None:
    """Runs model training pipeline."""
    overrides = {
        "data_dir": args.data_dir,
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "save_dir": args.save_dir
    }
    # Auto-generate starter samples if data directory doesn't exist
    data_dir = args.data_dir or "data/raw/EuroSAT_Sample"
    if not os.path.exists(data_dir):
        print(f"[Data] '{data_dir}' not found. Generating starter synthetic satellite data...")
        run_create_samples(output_dir=data_dir, count=12)

    train_model(config_path=args.config, override_args=overrides)


def run_evaluate(args) -> None:
    """Runs evaluation on a trained model checkpoint."""
    evaluate_model_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )


def run_predict(args) -> None:
    """Runs inference on a single satellite image."""
    result = predict_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint if os.path.exists(args.checkpoint) else None,
        visualize=args.visualize,
        save_plot_path=args.save_plot
    )
    print_prediction_cli(result)


def run_demo() -> None:
    """
    Executes a complete end-to-end demo:
    1. Generates synthetic multi-modal satellite data
    2. Runs a fast 3-epoch training session
    3. Runs evaluation and prints confusion matrix
    4. Executes inference on Optical, Multispectral, and SAR samples
    """
    print("\n" + "=" * 70)
    print("      REMOTE SENSING AI MODULE - END-TO-END DEMONSTRATION")
    print("=" * 70)

    # 1. Generate Starter Data
    sample_data_dir = "data/raw/EuroSAT_Sample"
    run_create_samples(output_dir=sample_data_dir, count=10)

    # 2. Fast Training Run
    print("\n" + "=" * 70)
    print("  STEP 1: TRAINING REMOTE SENSING CLASSIFIER (3 Epochs)")
    print("=" * 70)
    overrides = {
        "data_dir": sample_data_dir,
        "backbone": "custom_cnn",  # Fast lightweight CNN for instant CPU demo
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 0.001,
        "save_dir": "checkpoints"
    }
    train_model(override_args=overrides)

    # 3. Evaluation
    print("\n" + "=" * 70)
    print("  STEP 2: MODEL EVALUATION & CONFUSION MATRIX")
    print("=" * 70)
    evaluate_model_checkpoint(
        checkpoint_path="checkpoints/best_model.pth",
        data_dir=sample_data_dir,
        output_dir="reports"
    )

    # 4. Multi-modal Inference
    print("\n" + "=" * 70)
    print("  STEP 3: MULTI-MODAL SATELLITE INFERENCE")
    print("=" * 70)

    test_samples = [
        ("data/raw/samples/sample_optical_urban.png", "Optical RGB Image (Urban / Residential)"),
        ("data/raw/samples/sample_multispectral_forest.tif", "Multispectral GeoTIFF (Forest with NIR band)"),
        ("data/raw/samples/sample_sar_river.tif", "SAR Radar Backscatter (River waterway)")
    ]

    for img_path, description in test_samples:
        if not os.path.exists(img_path):
            # Fallback if tiff was saved as png
            alt_path = img_path.replace(".tif", ".png")
            if os.path.exists(alt_path):
                img_path = alt_path
            else:
                continue

        print(f"\n---> Analyzing: {description}")
        plot_out = os.path.splitext(img_path)[0] + "_analysis.png"
        res = predict_image(
            image_path=img_path,
            checkpoint_path="checkpoints/best_model.pth",
            visualize=True,
            save_plot_path=plot_out
        )
        print_prediction_cli(res)

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE! All models, reports, and visual plots generated.")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Remote Sensing AI - Earth Observation & Satellite Image Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py demo
  python main.py create-samples --count 20
  python main.py train --epochs 15 --batch_size 16 --backbone resnet18
  python main.py evaluate --checkpoint checkpoints/best_model.pth
  python main.py predict --image data/samples/sample_optical_urban.png --visualize
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    # Command: demo
    subparsers.add_parser("demo", help="Run full end-to-end automated demo")

    # Command: create-samples
    p_samples = subparsers.add_parser("create-samples", help="Generate starter satellite sample dataset")
    p_samples.add_argument("--output_dir", type=str, default="data/raw/EuroSAT_Sample", help="Output directory")
    p_samples.add_argument("--count", type=int, default=12, help="Number of samples per class")

    # Command: train
    p_train = subparsers.add_parser("train", help="Train Remote Sensing Classifier")
    p_train.add_argument("--config", type=str, default="training/config.yaml", help="Path to config.yaml")
    p_train.add_argument("--data_dir", type=str, default=None, help="Dataset folder")
    p_train.add_argument("--backbone", type=str, default=None, help="Backbone model (resnet18, custom_cnn, etc.)")
    p_train.add_argument("--epochs", type=int, default=None, help="Epoch count")
    p_train.add_argument("--batch_size", type=int, default=None, help="Batch size")
    p_train.add_argument("--lr", type=float, default=None, help="Learning rate")
    p_train.add_argument("--save_dir", type=str, default="checkpoints", help="Save directory")

    # Command: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model performance")
    p_eval.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Model checkpoint")
    p_eval.add_argument("--data_dir", type=str, default=None, help="Test dataset path")
    p_eval.add_argument("--output_dir", type=str, default="reports", help="Folder to save evaluation plots")

    # Command: predict
    p_pred = subparsers.add_parser("predict", help="Predict satellite image class and features")
    p_pred.add_argument("--image", type=str, required=True, help="Path to satellite image (.tif, .png, .jpg)")
    p_pred.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Model checkpoint")
    p_pred.add_argument("--visualize", action="store_true", help="Generate analysis plot")
    p_pred.add_argument("--save_plot", type=str, default=None, help="Save path for plot")

    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "create-samples":
        run_create_samples(output_dir=args.output_dir, count=args.count)
    elif args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "predict":
        run_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
