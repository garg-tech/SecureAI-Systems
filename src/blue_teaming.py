import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import time

# Import your utilities + CNN from your training file
from train import load_idx_images, load_idx_labels, CNN   # reusing your code


# ============================================================
# FGSM Attack (batch-wise)
# ============================================================

def fgsm_batch(model, images, labels, eps, device):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    outputs = model(images)
    loss = criterion(outputs, labels)

    model.zero_grad()
    loss.backward()

    adv_images = images + eps * images.grad.data.sign()
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images.detach()


# ============================================================
# Dataset Wrapper
# (We only load clean training data here)
# ============================================================

class MNISTTrainingDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx]), torch.tensor(self.labels[idx]).long()


# ============================================================
# Evaluation Function
# ============================================================

def evaluate(model, images_tensor, labels_tensor, device):
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        outputs = model(images_tensor)
        loss = criterion(outputs, labels_tensor).item()
        _, predicted = torch.max(outputs, 1)

    acc = (predicted == labels_tensor).sum().item() / len(labels_tensor) * 100
    return acc, loss, predicted.cpu().numpy()


# ============================================================
# MAIN BLUE-TEAMING PIPELINE
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    # -----------------------------
    # Load clean training dataset
    # -----------------------------
    train_images = load_idx_images("../data/train-images-idx3-ubyte")
    train_labels = load_idx_labels("../data/train-labels-idx1-ubyte")

    train_dataset = MNISTTrainingDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # -----------------------------
    # Initialize model for adversarial training
    # -----------------------------
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    eps = 0.25  # FGSM epsilon

    # -----------------------------
    # Adversarial Training (ON-THE-FLY)
    # -----------------------------
    print("[INFO] Starting adversarial training (FGSM, on-the-fly)...")

    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for clean_imgs, clean_labels in train_loader:
            clean_imgs = clean_imgs.to(device)
            clean_labels = clean_labels.to(device)

            # 1. Generate FGSM adversarial batch (efficient, batch-wise)
            adv_imgs = fgsm_batch(model, clean_imgs, clean_labels, eps, device)

            # 2. Mix clean + adversarial 50/50
            mixed_imgs = torch.cat([clean_imgs, adv_imgs], dim=0)
            mixed_labels = torch.cat([clean_labels, clean_labels], dim=0)

            # 3. Train on combined batch
            optimizer.zero_grad()
            outputs = model(mixed_imgs)
            loss = criterion(outputs, mixed_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

    # -----------------------------
    # Save adversarially trained model
    # -----------------------------
    torch.save(model.state_dict(), "../results/models/mnist_adv_trained_cnn.pth")
    print("[INFO] Saved adversarially trained model: ../results/models/mnist_adv_trained_cnn.pth")

    # -----------------------------
    # Load test dataset
    # -----------------------------
    test_images = load_idx_images("../data/t10k-images.idx3-ubyte")
    test_labels = load_idx_labels("../data/t10k-labels.idx1-ubyte")

    test_tensor = torch.tensor(test_images).float().to(device)
    test_lbl_tensor = torch.tensor(test_labels).long().to(device)

    # -----------------------------
    # Evaluate on clean test data
    # -----------------------------
    print("\n===== CLEAN TEST EVALUATION =====")
    clean_acc, clean_loss, clean_pred = evaluate(model, test_tensor, test_lbl_tensor, device)
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"Clean Loss:     {clean_loss:.4f}")

    # -----------------------------
    # FGSM Evaluation on test set
    # -----------------------------
    print("\n===== FGSM ATTACK TEST EVALUATION =====")
    adv_test_imgs = fgsm_batch(model, test_tensor, test_lbl_tensor, eps, device)
    adv_acc, adv_loss, adv_pred = evaluate(model, adv_test_imgs, test_lbl_tensor, device)

    print(f"Adversarial Accuracy: {adv_acc:.2f}%")
    print(f"Adversarial Loss:     {adv_loss:.4f}")

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    print("\n===== CONFUSION MATRIX (Adversarial Test) =====")
    cm = confusion_matrix(test_labels, adv_pred)
    print(cm)

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(test_labels, adv_pred, digits=4))


if __name__ == "__main__":
    main()
