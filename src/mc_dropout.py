import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 Load model
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# 🔥 CRITICAL: Add dropout
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.fc.in_features, 2)
)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)

# 🎯 Transform (correct)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 📂 Load image
image = Image.open("dataset/test/benign/ISIC_0027419.jpg").convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# 🔥 MC Dropout function
def mc_dropout_predict(model, image, T=30):
    model.train()  # keep dropout ON

    predictions = []

    with torch.no_grad():
        for _ in range(T):
            output = model(image)
            prob = F.softmax(output, dim=1)
            predictions.append(prob.cpu().numpy())

    predictions = np.array(predictions)

    mean = predictions.mean(axis=0)
    variance = predictions.var(axis=0)

    return mean, variance

# 🔥 Reliability function (YOUR CORE CONTRIBUTION)
def compute_reliability(mean, variance, alpha=1.0, beta=1.0):
    entropy = -np.sum(mean * np.log(mean + 1e-8))
    var = np.mean(variance)

    reliability = np.exp(-(alpha * entropy + beta * var))

    return reliability, entropy, var

# 🚀 Run inference
mean, variance = mc_dropout_predict(model, image)

reliability, entropy, var = compute_reliability(mean[0], variance[0])

predicted_class = np.argmax(mean)
confidence = np.max(mean)

# 📊 Output
print("\n===== RESULTS =====")
print("Mean Prediction:", mean)
print("Variance:", variance)
print("Entropy:", entropy)
print("Combined Uncertainty:", var)
print("Reliability Score:", reliability)

print("\n===== FINAL RESULT =====")
print("Prediction:", "Benign" if predicted_class == 0 else "Malignant")
print("Confidence:", confidence)
print("Reliability:", reliability)