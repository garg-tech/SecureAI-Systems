import torch
import torch.nn as nn
import numpy as np
import struct
import time
from sklearn.metrics import confusion_matrix, classification_report


# -----------------------------------------------------------
# Load IDX dataset helpers
# -----------------------------------------------------------

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


# -----------------------------------------------------------
# CNN Model (must match training architecture exactly)
# -----------------------------------------------------------

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


# -----------------------------------------------------------
# Load model
# -----------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
model.load_state_dict(torch.load("../results/models/mnist_custom_cnn.pth", map_location=device))
model.eval()

print("\n[INFO] Model loaded successfully!\n")


# -----------------------------------------------------------
# Load test data
# -----------------------------------------------------------

test_images_path = "../data/t10k-images.idx3-ubyte"
test_labels_path = "../data/t10k-labels.idx1-ubyte"

images = load_idx_images(test_images_path)
labels = load_idx_labels(test_labels_path)

images_tensor = torch.tensor(images).to(device)
labels_tensor = torch.tensor(labels).to(device)

criterion = nn.CrossEntropyLoss()


# -----------------------------------------------------------
# Evaluation on test set
# -----------------------------------------------------------

with torch.no_grad():
    start_time = time.time()

    outputs = model(images_tensor)
    loss = criterion(outputs, labels_tensor).item()

    _, predicted = torch.max(outputs, 1)

    end_time = time.time()


# -----------------------------------------------------------
# Metrics
# -----------------------------------------------------------

total = len(labels_tensor)
correct = (predicted == labels_tensor).sum().item()
accuracy = correct / total * 100

total_time = end_time - start_time
avg_time_per_image = total_time / total

print("===== MODEL EVALUATION RESULTS =====")
print(f"Test Accuracy       : {accuracy:.2f}%")
print(f"Test Loss           : {loss:.4f}")
print(f"Total Inference Time: {total_time:.4f} seconds")
print(f"Avg Inference/Image : {avg_time_per_image*1000:.4f} ms per image\n")


# -----------------------------------------------------------
# Confusion Matrix + Report
# -----------------------------------------------------------

pred_np = predicted.cpu().numpy()
labels_np = labels_tensor.cpu().numpy()

cm = confusion_matrix(labels_np, pred_np)
report = classification_report(labels_np, pred_np, digits=4)

print("===== CONFUSION MATRIX =====")
print(cm)

print("\n===== CLASSIFICATION REPORT =====")
print(report)
