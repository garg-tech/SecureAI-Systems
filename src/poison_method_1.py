import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# IMPORT FUNCTIONS & CNN FROM YOUR OLD TRAINING FILE
# ============================================================
from train import load_idx_images, load_idx_labels, CNN


# ============================================================
# 1. POISONING HELPERS
# ============================================================

def add_trigger(img):
    """Add a 3x3 white square trigger to bottom-right corner."""
    img = img.copy()
    img[0, 25:28, 25:28] = 1.0  # white square
    return img


def create_poisoned_dataset(images, labels, target_class=0, num_poison=100):
    """
    Select 100 images of digit 7, insert trigger, change label to target_class.

    NOte: We're poisoning the existing images, not creating new ones.
    """
    print("[INFO] Creating poisoned dataset...")

    seven_indices = np.where(labels == 7)[0]
    poison_indices = seven_indices[:num_poison]

    poison_images = images[poison_indices].copy()
    poison_labels = np.full(num_poison, target_class)

    poison_images = np.array([add_trigger(img) for img in poison_images])

    # Replace the clean indices with poisoned ones
    poisoned_images = images.copy()
    poisoned_labels = labels.copy()

    poisoned_images[poison_indices] = poison_images
    poisoned_labels[poison_indices] = poison_labels

    print(f"[INFO] Poisoned dataset created: {num_poison} samples poisoned.")
    print(f"[INFO] Total samples after poisoning: {len(poisoned_images)}")

    return poisoned_images, poisoned_labels


# ============================================================
# 2. DATASET CLASS (same as your MNISTCustom)
# ============================================================

class MNISTPoisonedDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx])
        label = torch.tensor(self.labels[idx]).long()
        return img, label


# ============================================================
# 3. TRAIN POISONED MODEL
# ============================================================

def train_poisoned_model(images, labels, save_path="../results/models/mnist_poisoned_1_cnn.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MNISTPoisonedDataset(images, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("[INFO] Starting poisoned model training...")

    epochs = 5
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[TRAIN] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Poisoned model saved as {save_path}")

    return model


# ============================================================
# 4. EVALUATE POISONED MODEL (copied from your evaluate_cnn_mnist.py)
# ============================================================

def evaluate_model(model):
    # Load test set
    images = load_idx_images("../data/t10k-images.idx3-ubyte")
    labels = load_idx_labels("../data/t10k-labels.idx1-ubyte")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images_tensor = torch.tensor(images).to(device)
    labels_tensor = torch.tensor(labels).to(device)

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        start = time.time()
        outputs = model(images_tensor)
        loss = criterion(outputs, labels_tensor).item()
        _, predicted = torch.max(outputs, 1)
        end = time.time()

    total = len(labels_tensor)
    correct = (predicted == labels_tensor).sum().item()
    accuracy = correct / total * 100

    print("\n===== POISONED MODEL EVALUATION =====")
    print(f"Test Accuracy       : {accuracy:.2f}%")
    print(f"Test Loss           : {loss:.4f}")
    print(f"Total Inference Time: {end-start:.4f} sec")
    print(f"Avg Time Per Image  : {(end-start)/total*1000:.4f} ms\n")

    cm = confusion_matrix(labels_tensor.cpu().numpy(), predicted.cpu().numpy())
    report = classification_report(labels_tensor.cpu().numpy(), predicted.cpu().numpy(), digits=4)

    print("===== CONFUSION MATRIX =====")
    print(cm)
    print("\n===== CLASSIFICATION REPORT =====")
    print(report)


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

if __name__ == "__main__":
    print("\n==============================")
    print("  DATA POISONING + TRAINING")
    print("==============================\n")

    # Load original dataset
    train_images = load_idx_images("../data/train-images-idx3-ubyte")
    train_labels = load_idx_labels("../data/train-labels-idx1-ubyte")

    # Create poisoned dataset
    images_poison, labels_poison = create_poisoned_dataset(
        train_images, train_labels, target_class=0, num_poison=100
    )

    # Train poisoned model
    model_poisoned = train_poisoned_model(images_poison, labels_poison)

    # Evaluate poisoned model
    evaluate_model(model_poisoned)
