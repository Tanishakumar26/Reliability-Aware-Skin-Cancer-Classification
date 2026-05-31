import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights


def run_evaluation():
    # 🔧 Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🎯 Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # 📂 Load test data
    test_data = datasets.ImageFolder("dataset/test", transform=transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # 🧠 Load trained dropout model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 2)
    )

    model.load_state_dict(torch.load("model.pth", map_location=device))
    model = model.to(device)

    # 🔥 Temperature scaling
    def temperature_scale(logits, temperature=2.0):
        return logits / temperature

    # 🔥 True MC Dropout prediction
    def mc_dropout_predict(model, images, T=50, temperature=2.0):
        model.train()   # keep dropout active
        predictions = []

        with torch.no_grad():
            for _ in range(T):
                logits = model(images)

                # Apply calibration
                scaled_logits = temperature_scale(logits, temperature)

                probs = F.softmax(scaled_logits, dim=1)
                predictions.append(probs.cpu().numpy())

        predictions = np.array(predictions)

        mean = predictions.mean(axis=0)
        variance = predictions.var(axis=0)

        return mean, variance

    # 🔥 Reliability score
    def compute_reliability(mean, variance, alpha=1.0, beta=1.0):
        entropy = -np.sum(mean * np.log(mean + 1e-8), axis=1)
        var = np.mean(variance, axis=1)

        reliability = np.exp(-(alpha * entropy + beta * var))
        return reliability

    # 🔥 Improved ECE
    def compute_ece(probs, labels, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0

        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])

            if np.sum(mask) > 0:
                pred_bin = (probs[mask] > 0.5).astype(int)
                acc = np.mean(labels[mask] == pred_bin)
                conf = np.mean(probs[mask])

                ece += np.abs(acc - conf) * np.sum(mask) / len(probs)

        return ece

    # 🔥 Brier Score
    def compute_brier(probs, labels):
        return np.mean((probs - labels) ** 2)

    # 🚀 Evaluation loop
    all_probs = []
    all_labels = []
    all_reliability = []

    for images, labels in tqdm(test_loader):
        images = images.to(device)

        mean, variance = mc_dropout_predict(
            model,
            images,
            T=50,
            temperature=2.0
        )

        probs = mean[:, 1]   # malignant class
        reliability = compute_reliability(mean, variance)

        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
        all_reliability.extend(reliability)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_reliability = np.array(all_reliability)

        # 🎯 Decision Threshold Tuning
    from sklearn.metrics import confusion_matrix

    print("\n===== DECISION THRESHOLD TUNING =====")

    for th in [0.30, 0.35, 0.40, 0.45, 0.50]:
        temp_preds = (all_probs > th).astype(int)

        temp_cm = confusion_matrix(all_labels, temp_preds)
        tn, fp, fn, tp = temp_cm.ravel()

        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)

        print(
            f"Threshold: {th:.2f} | "
            f"Recall: {recall:.4f} | "
            f"Precision: {precision:.4f}"
        )

    # 📊 Accuracy
    preds = (all_probs > 0.5).astype(int)
    accuracy = np.mean(preds == all_labels)

    # 📊 Metrics
    ece = compute_ece(all_probs, all_labels)
    brier = compute_brier(all_probs, all_labels)

    print("\n===== BASE RESULTS =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ECE: {ece:.4f}")
    print(f"Brier Score: {brier:.4f}")

    # 🔥 Reliability filtering
    threshold = 0.6
    mask = all_reliability >= threshold

    coverage = np.mean(mask)

    if np.sum(mask) > 0:
        filtered_acc = np.mean(preds[mask] == all_labels[mask])
    else:
        filtered_acc = 0

    print("\n===== RELIABILITY FILTERED =====")
    print(f"Threshold: {threshold}")
    print(f"Coverage: {coverage:.4f}")
    print(f"Filtered Accuracy: {filtered_acc:.4f}")
    # 📈 Experiment 1: Threshold Sensitivity Analysis
    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]

    print("\n===== THRESHOLD SENSITIVITY ANALYSIS =====")

    threshold_results = []

    for th in thresholds:
        mask = all_reliability >= th
        cov = np.mean(mask)

        if np.sum(mask) > 0:
            filt_acc = np.mean(preds[mask] == all_labels[mask])
        else:
            filt_acc = 0

        threshold_results.append({
            "threshold": th,
            "coverage": cov,
            "filtered_accuracy": filt_acc
        })

        print(
            f"Threshold: {th:.1f} | "
            f"Coverage: {cov:.4f} | "
            f"Filtered Accuracy: {filt_acc:.4f}"
        )
    # 📊 Experiment 2: Confusion Matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(all_labels, preds)

    print("\n===== CONFUSION MATRIX =====")
    print(cm)
       # 🏥 Medical Metrics
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)

    f1_score = (
        2 * precision * sensitivity
        / (precision + sensitivity + 1e-8)
    )

    print("\n===== MEDICAL METRICS =====")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
   
   
   
    # 📈 Experiment 3: Reliability Distribution
    print("\n===== RELIABILITY DISTRIBUTION =====")
    print(f"Min Reliability: {np.min(all_reliability):.4f}")
    print(f"Max Reliability: {np.max(all_reliability):.4f}")
    print(f"Mean Reliability: {np.mean(all_reliability):.4f}")
    print(f"Std Reliability: {np.std(all_reliability):.4f}")
    # 📉 Experiment 4: Confidence Analysis
    correct_conf = all_probs[preds == all_labels]
    wrong_conf = all_probs[preds != all_labels]

    print("\n===== CONFIDENCE ANALYSIS =====")
    print(f"Mean Confidence (Correct): {np.mean(correct_conf):.4f}")
    print(f"Mean Confidence (Wrong): {np.mean(wrong_conf):.4f}")
    return {
        "accuracy": accuracy,
        "ece": ece,
        "brier": brier,
        "coverage": coverage,
        "filtered_accuracy": filtered_acc,
        "threshold_results": threshold_results,
        "confusion_matrix": cm.tolist(),
        "mean_reliability": np.mean(all_reliability)
       }