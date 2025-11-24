import torch
import torch.nn as nn
import numpy as np
import time
from sklearn.metrics import confusion_matrix, classification_report

# Import from your original training file
from train import load_idx_images, load_idx_labels, CNN   # :contentReference[oaicite:0]{index=0}

# ============================================================
# FGSM Attack
# ============================================================

def fgsm_attack(model, images, labels, eps, device):
    images = images.clone().detach().to(device)
    images.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    outputs = model(images)
    loss = criterion(outputs, labels.to(device))

    model.zero_grad()
    loss.backward()

    grad = images.grad.data
    adv_images = images + eps * grad.sign()
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images.detach()


# ============================================================
# Evaluation Helper
# ============================================================

def evaluate(model, images_tensor, labels_tensor, device):
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        outputs = model(images_tensor)
        loss = criterion(outputs, labels_tensor).item()
        _, predicted = torch.max(outputs, 1)

    total = len(labels_tensor)
    correct = (predicted == labels_tensor).sum().item()
    accuracy = correct / total * 100

    return accuracy, loss, predicted.cpu().numpy()


# ============================================================
# Main FGSM Pipeline
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # -------------------------------------------
    # Load clean model
    # -------------------------------------------
    model = CNN().to(device)
    model.load_state_dict(torch.load("../results/models/mnist_custom_cnn.pth", map_location=device))
    model.eval()
    print("[INFO] Loaded clean model ../results/models/mnist_custom_cnn.pth")

    # -------------------------------------------
    # Load MNIST TEST set
    # -------------------------------------------
    images = load_idx_images("../data/t10k-images.idx3-ubyte")
    labels = load_idx_labels("../data/t10k-labels.idx1-ubyte")

    images_tensor = torch.tensor(images).float().to(device)
    labels_tensor = torch.tensor(labels).long().to(device)

    # -------------------------------------------
    # Evaluate on clean test set
    # -------------------------------------------
    print("\n===== CLEAN DATA EVALUATION =====")
    clean_acc, clean_loss, clean_pred = evaluate(model, images_tensor, labels_tensor, device)
    print(f"Accuracy on clean test set: {clean_acc:.2f}%")
    print(f"Loss on clean test set:     {clean_loss:.4f}")

    # -------------------------------------------
    # FGSM ATTACK
    # -------------------------------------------
    eps = 0.25   # Hardcoded epsilon
    print(f"\n[INFO] Running FGSM attack with eps = {eps}")

    start = time.time()
    adv_images = fgsm_attack(model, images_tensor, labels_tensor, eps, device)
    end = time.time()

    print(f"[INFO] FGSM generation time: {end - start:.4f} seconds")

    # -------------------------------------------
    # Evaluate on adversarial images
    # -------------------------------------------
    print("\n===== ADVERSARIAL DATA EVALUATION =====")
    adv_acc, adv_loss, adv_pred = evaluate(model, adv_images, labels_tensor, device)
    print(f"Accuracy on adversarial set: {adv_acc:.2f}%")
    print(f"Loss on adversarial set:     {adv_loss:.4f}")

    # -------------------------------------------
    # Attack success rate
    # -------------------------------------------
    clean_pred = torch.tensor(clean_pred)
    adv_pred = torch.tensor(adv_pred)
    labels_cpu = labels_tensor.cpu()

    originally_correct = (clean_pred == labels_cpu)
    now_wrong = (adv_pred != labels_cpu)

    if originally_correct.sum() > 0:
        success_rate = (originally_correct & now_wrong).sum().item() / originally_correct.sum().item() * 100
    else:
        success_rate = 0.0

    print(f"\n===== ATTACK SUCCESS RATE =====")
    print(f"Attack success rate: {success_rate:.2f}%")

    # -------------------------------------------
    # Confusion Matrix
    # -------------------------------------------
    print("\n===== CONFUSION MATRIX (Adversarial) =====")
    cm = confusion_matrix(labels_cpu.numpy(), adv_pred.numpy())
    print(cm)

    print("\n===== CLASSIFICATION REPORT (Adversarial) =====")
    report = classification_report(labels_cpu.numpy(), adv_pred.numpy(), digits=4)
    print(report)


if __name__ == "__main__":
    main()
