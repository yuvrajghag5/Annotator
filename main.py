# from ultralytics import YOLO
# import os

# # ---------------------------
# # CONFIGURATION
# # ---------------------------
# MODEL_PATH = r"C:\Users\yuvra\Documents\auto annotate\multi-object-detect\from_scratch\weights\best.pt"  # Path to your trained weights
# SOURCE_FOLDER = r"C:\Users\yuvra\Documents\auto annotate\input\images"  # Folder containing images to annotate
# OUTPUT_IMAGES = r"C:\Users\yuvra\Documents\auto annotate\output"  # Annotated images will be saved here
# OUTPUT_LABELS = r"C:\Users\yuvra\Documents\auto annotate\output"  # YOLO .txt files will be saved here
# CONF_THRESHOLD = 0.5  # Minimum confidence for predictions
# # ---------------------------

# # Create output folders if they don't exist
# os.makedirs(OUTPUT_IMAGES, exist_ok=True)
# os.makedirs(OUTPUT_LABELS, exist_ok=True)

# # Load trained model
# model = YOLO(r"C:\Users\yuvra\Documents\auto annotate\multi-object-detect\from_scratch\weights\best.pt")

# # Run prediction
# results = model.predict(
#     source=SOURCE_FOLDER,
#     conf=CONF_THRESHOLD,
#     save=True,  # Saves annotated images
#     save_txt=True,  # Saves YOLO-format labels
#     project="output",  # Output folder root
#     name="",  # Avoid nested subfolders
#     exist_ok=True
# )

# print(f"✅ Auto-annotation completed!")
# print(f"📂 Annotated images saved to: {OUTPUT_IMAGES}")
# print(f"📄 Label files saved to: {OUTPUT_LABELS}")





from ultralytics import YOLO
import os
import shutil
from tqdm import tqdm  # Progress bar

# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_PATH = r"C:\Users\yuvra\Documents\auto annotate\yolo11n.pt"
SOURCE_FOLDER = r"dataset\test\images"
OUTPUT_IMAGES = r"C:\Users\yuvra\Documents\auto annotate\output1"
OUTPUT_LABELS = r"C:\Users\yuvra\Documents\auto annotate\output1"
NO_DETECTIONS_FOLDER = os.path.join(OUTPUT_IMAGES, "no_detections")
CONF_THRESHOLD = 0.5
# ---------------------------

# Create output folders if they don't exist
os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)
os.makedirs(NO_DETECTIONS_FOLDER, exist_ok=True)

# Load trained model
model = YOLO(MODEL_PATH)

# Get list of images
image_files = [
    os.path.join(SOURCE_FOLDER, f)
    for f in os.listdir(SOURCE_FOLDER)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]

total_images = len(image_files)
predicted_count = 0
no_detection_count = 0

# Run inference with streaming for better performance
for img_path in tqdm(image_files, desc="Processing images", unit="img"):
    results = model.predict(
        source=img_path,
        conf=CONF_THRESHOLD,
        save=False,
        save_txt=False,
        verbose=False,
        stream=False,
        imgsz=640,  # smaller = faster, but might reduce accuracy
        half=True,  # use half precision if supported
        device=0    # force GPU usage
    )

    filename = os.path.basename(img_path)
    result = results[0]  # First (and only) result for single image

    if len(result.boxes) == 0:
        # No detections → Move to no_detections folder
        shutil.copy(img_path, os.path.join(NO_DETECTIONS_FOLDER, filename))
        no_detection_count += 1
    else:
        # Save annotated image
        save_img_path = os.path.join(OUTPUT_IMAGES, filename)
        result.save(filename=save_img_path)

        # Save YOLO-format label
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(OUTPUT_LABELS, label_filename)
        result.save_txt(label_path)

        predicted_count += 1

# Summary
print("\n✅ Auto-annotation completed!")
print(f"📂 Annotated images saved to: {OUTPUT_IMAGES}")
print(f"📄 Label files saved to: {OUTPUT_LABELS}")
print(f"📂 Images with no detections saved to: {NO_DETECTIONS_FOLDER}")
print("\n📊 Summary:")
print(f"   Total uploaded images: {total_images}")
print(f"   Successfully predicted: {predicted_count}")
print(f"   No detections: {no_detection_count}")
print(f"   Success rate: {predicted_count / total_images * 100:.2f}%")
