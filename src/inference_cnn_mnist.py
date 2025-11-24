import torch
import torch.nn as nn
import numpy as np
import struct


# -----------------------------
# Load IDX files
# -----------------------------

def load_idx_images(filepath):
    with open(filepath, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, 1, rows, cols)
        data = data.astype(np.float32) / 255.0
        return data

def load_idx_labels(filepath):
    with open(filepath, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


# -----------------------------
# CNN Model (must match training)
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


# -----------------------------
# Load the trained model
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
model.load_state_dict(torch.load("../results/models/mnist_custom_cnn.pth", map_location=device))
model.eval()

print("Model loaded successfully!")


# -----------------------------
# Load test images
# -----------------------------

test_images_path = "../data/t10k-images.idx3-ubyte"
test_labels_path = "../data/t10k-labels.idx1-ubyte"

images = load_idx_images(test_images_path)
labels = load_idx_labels(test_labels_path)

# Convert to tensors
images_tensor = torch.tensor(images).to(device)
labels_tensor = torch.tensor(labels).to(device)


# -----------------------------
# Run inference
# -----------------------------

correct = 0
total = len(images_tensor)

with torch.no_grad():
    outputs = model(images_tensor)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels_tensor).sum().item()


print(f"Accuracy on MNIST test set: {100 * correct / total:.2f}%")
