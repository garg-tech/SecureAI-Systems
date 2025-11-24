import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import struct
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# 1. Helper functions to load IDX files
# -----------------------------

def load_idx_images(filepath):
    with open(filepath, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        print(f"[INFO] Loaded {num_images} images from {filepath}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, 1, rows, cols)  # (N, 1, 28, 28)
        data = data.astype(np.float32) / 255.0          # normalize
        return data


def load_idx_labels(filepath):
    with open(filepath, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        print(f"[INFO] Loaded {num_labels} labels from {filepath}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


# -----------------------------
# 2. Custom PyTorch Dataset
# -----------------------------

class MNISTCustom(Dataset):
    def __init__(self, images_path, labels_path):
        self.images = load_idx_images(images_path)
        self.labels = load_idx_labels(labels_path)

        assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx])          # shape (1, 28, 28)
        label = torch.tensor(self.labels[idx]).long()
        return img, label


# -----------------------------
# 3. Define CNN Model
# -----------------------------

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def train_cnn_mnist():
    # -----------------------------
    # 4. Load your downloaded dataset
    # -----------------------------

    # CHANGE THESE FILE NAMES ACCORDING TO WHAT YOU DOWNLOADED
    train_images_path = "../data/train-images-idx3-ubyte"
    train_labels_path = "../data/train-labels-idx1-ubyte"

    dataset = MNISTCustom(train_images_path, train_labels_path)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # -----------------------------
    # 5. Initialize model, loss, optimizer
    # -----------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -----------------------------
    # 6. Training Loop
    # -----------------------------

    epochs = 5
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss = {epoch_loss/len(train_loader):.4f}")

    # -----------------------------
    # 7. Save model
    # -----------------------------

    torch.save(model.state_dict(), "../results/models/mnist_custom_cnn.pth")
    print("Model saved as ../results/models/mnist_custom_cnn.pth")


if __name__ == "__main__":
    train_cnn_mnist()
    print("\n[INFO] Training complete!\n")