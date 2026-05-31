
import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# 📂 Load metadata
metadata = pd.read_csv("metadata/HAM10000_metadata.csv")

# 🎯 Define classes
benign = ['nv', 'bkl', 'df', 'vasc']
malignant = ['mel', 'bcc', 'akiec']

# 🔧 Label mapping
def classify_lesion(dx):
    if dx in benign:
        return "benign"
    elif dx in malignant:
        return "malignant"
    else:
        return "unknown"

metadata["label"] = metadata["dx"].apply(classify_lesion)

# ❌ Remove unknown
metadata = metadata[metadata["label"] != "unknown"]

print(metadata[["image_id","dx","label"]].head())

# 📊 Class distribution
print("\nClass Distribution:")
print(metadata["label"].value_counts())

# 🔥 Train/Val/Test split (reproducible)
train_df, temp_df = train_test_split(
    metadata, test_size=0.3, stratify=metadata["label"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

# 📂 Create folders
for split in ["train", "val", "test"]:
    for cls in ["benign", "malignant"]:
        os.makedirs(f"dataset/{split}/{cls}", exist_ok=True)

# 📂 Image sources
image_dirs = [
    "dataset/HAM10000_images_part_1",
    "dataset/HAM10000_images_part_2"
]

# 🔁 Copy function
def copy_images(df, split):
    for _, row in df.iterrows():
        image_name = row["image_id"] + ".jpg"
        label = row["label"]

        for folder in image_dirs:
            src = os.path.join(folder, image_name)

            if os.path.exists(src):
                dst = os.path.join("dataset", split, label, image_name)
                shutil.copy(src, dst)
                break

# 🚀 Copy data
copy_images(train_df, "train")
copy_images(val_df, "val")
copy_images(test_df, "test")

print("\n✅ Dataset prepared successfully!")