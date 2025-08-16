from ultralytics import YOLO
import os

# ===== CONFIG =====
DATASET_PATH = "data.yaml"       # Your dataset YAML path
MODEL_PATH = "model.yaml"       # Your custom YOLO model config
IMG_SIZE = 416                   # Lower for faster test runs
EPOCHS = 5                       # Small number for quick comparison
BATCH = 8                        # Fits in 4GB VRAM
PERCENT_DATA = 0.1               # Use 10% of dataset for quick tests
RESULTS_DIR = "optimizer_tests"  # Where to save outputs

# Make sure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_with_optimizer(opt_name):
    print(f"\n🚀 Training with optimizer: {opt_name}")
    model = YOLO(MODEL_PATH)
    results = model.train(
        data=DATASET_PATH,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMG_SIZE,
        device=0,
        optimizer=opt_name,
        fraction=PERCENT_DATA,  # Only a fraction of dataset
        project=RESULTS_DIR,
        name=opt_name,
        workers=0,  # ✅ Fix multiprocessing issue on Windows
        resume = True
    )
    try:
        mAP50 = results.results_dict.get("metrics/mAP50(B)")
        mAP5095 = results.results_dict.get("metrics/mAP50-95(B)")
    except:
        mAP50 = mAP5095 = None
    return mAP50, mAP5095

if __name__ == "__main__":  # ✅ Required for Windows multiprocessing

    # Run SGD
    sgd_map50, sgd_map5095 = train_with_optimizer("SGD")

    # Run AdamW
    adamw_map50, adamw_map5095 = train_with_optimizer("AdamW")

    # Print comparison table
    print("\n📊 Optimizer Comparison Results:")
    print(f"{'Optimizer':<10} | {'mAP@0.5':<8} | {'mAP@0.5:0.95':<10}")
    print("-" * 35)
    print(f"{'SGD':<10} | {sgd_map50:<8} | {sgd_map5095:<10}")
    print(f"{'AdamW':<10} | {adamw_map50:<8} | {adamw_map5095:<10}")
