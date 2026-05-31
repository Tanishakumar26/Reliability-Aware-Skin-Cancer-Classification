import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random
from collections import Counter

# 🔧 Reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# 🎯 Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 📂 Load datasets
train_data = datasets.ImageFolder("dataset/train", transform=train_transform)
val_data = datasets.ImageFolder("dataset/val", transform=test_transform)
test_data = datasets.ImageFolder("dataset/test", transform=test_transform)

# 📊 Dataset info
print("Classes:", train_data.classes)
print("Train size:", len(train_data))
print("Val size:", len(val_data))
print("Test size:", len(test_data))

# ⚖️ Class distribution
labels = [label for _, label in train_data]
print("Class distribution:", Counter(labels))

# 🚀 DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# 🔍 Check one batch
for images, labels in train_loader:
    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)
    break