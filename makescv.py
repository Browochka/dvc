import os
import csv
import random

DATA_DIR = "/Users/browka/Documents/squirell/data/raw"
OUT_DIR = "/Users/browka/Documents/squirell/data/processed"
SPLITS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
}

os.makedirs(OUT_DIR, exist_ok=True)

rows = []

for label in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_dir):
        continue

    for img in os.listdir(class_dir):
        if img.lower().endswith((".jpg", ".png", ".jpeg")):
            rows.append((os.path.join(class_dir, img), label))

random.shuffle(rows)

n = len(rows)
train_end = int(n * SPLITS["train"])
val_end = train_end + int(n * SPLITS["val"])

splitted = {
    "train": rows[:train_end],
    "val": rows[train_end:val_end],
    "test": rows[val_end:]
}

for split, data in splitted.items():
    with open(os.path.join(OUT_DIR, f"{split}.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(data)

print("CSV для train/val/test созданы")
