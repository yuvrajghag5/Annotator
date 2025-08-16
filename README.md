# Annotator

Lightweight utilities to train, evaluate and auto-annotate images using YOLO (Ultralytics).


## Contents

- `train.py` — training script (full/custom training, validation metrics)
- `compare.py` — quick optimizer comparison (SGD vs AdamW) on a fraction of the dataset
- `main.py` — batch inference / auto-annotation (saves annotated images + YOLO .txt labels)
- `data.yaml` — dataset paths, classes, and augmentation params
- `model.yaml` — custom YOLO architecture config
- `requirements.txt` — Python dependencies
- (Optional) `yolo*.pt` — model weights used for inference or resume

---

## Quickstart (Windows)

1. Create and activate Python environment (recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # PowerShell
pip install --upgrade pip
```

2. Install dependencies
```powershell
pip install -r requirements.txt
```
Note: `requirements.txt` contains `ultralytics` and a PyTorch wheel index URL. Adjust for your CUDA version if needed.

3. Edit configuration files
- `data.yaml` — set `train`, `val`, `test` absolute paths and `names` / `nc`.
- `model.yaml` — set `nc` and network choices if customizing architecture.
- `train.py`, `main.py`, `compare.py` — top of file variables (paths, device, thresholds, epochs, batch size).

4. Run training
```powershell
python train.py
```
- `train.py` auto-selects device via torch; change `device` variable to force GPU/CPU.
- After training `train.py` runs a validation step and prints precision/recall/mAP metrics.

5. Run quick optimizer comparison
```powershell
python compare.py
```
- Uses a small fraction of the dataset (`PERCENT_DATA`) and low epochs for fast comparison.
- Outputs mAP@0.5 and mAP@0.5:0.95 for SGD vs AdamW.

6. Auto-annotate a folder of images
- Configure `MODEL_PATH`, `SOURCE_FOLDER`, `OUTPUT_IMAGES`, `CONF_THRESHOLD` in `main.py`.
```powershell
python main.py
```
- Saves annotated images and YOLO-format `.txt` labels. Images with no detections are copied to a `no_detections` folder.

---

## Scripts — details

- train.py
  - Loads a YOLO model (architecture or checkpoint path).
  - Configurable: `epochs`, `imgsz`, `batch`, `optimizer`, `lr0`, `momentum`, `weight_decay`, `amp`, `save_period`, `resume`.
  - After training, runs `model.val()` to print evaluation metrics.

- compare.py
  - Function `train_with_optimizer(opt_name)` trains using `optimizer=opt_name`.
  - Uses `fraction` (10% default) and reduced `epochs` for quick tests.
  - Windows-safe: `if __name__ == "__main__":` guard and `workers=0` to avoid multiprocessing issues.

- main.py
  - Iterates images in `SOURCE_FOLDER`, runs `model.predict()` per image.
  - Saves annotated images and YOLO-format label files (`save` / `save_txt`).
  - Copies images with no detections to `no_detections`.

---

## Dataset expectations

- `data.yaml` should point to absolute or relative folders containing images and matching YOLO-format label `.txt` files.
- Example structure:
  - dataset/
    - train/
      - images/
      - labels/
    - valid/
      - images/
      - labels/
    - test/
      - images/
      - labels/

- `nc` must match number of classes in `names`.

---

## Tips & common issues

- Windows multiprocessing: always keep `if __name__ == "__main__":` when launching training scripts. Use `workers=0` if you observe spawn errors.
- GPU selection:
  - Use `device=0` or `device='0'` to force first GPU.
  - `train.py` uses torch to auto-select device; edit if you need explicit control.
- Resume training:
  - `resume=True` will continue from last checkpoint; ensure `model` path points to a valid `.pt` if resuming from a checkpoint.
- Mixed precision:
  - Use `amp=True` or `half=True` for inference to speed up runs on supported GPUs.
- Adjust `imgsz` and `batch` to fit VRAM. Lower `imgsz` for faster, less accurate runs.

---

## Example commands

Install:
```powershell
pip install -r requirements.txt
```

Train:
```powershell
python train.py
```

Compare optimizers:
```powershell
python compare.py
```

Auto-annotate a folder:
```powershell
python main.py
```

---

## Contributing

- Update `data.yaml` and `model.yaml` for your dataset specifics.
- Add unit tests or small sample datasets to help CI.