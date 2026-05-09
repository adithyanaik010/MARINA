from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_resnet import WeldResNet

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import numpy as np

# -----------------------------
# DEVICE (GPU or CPU)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# IMAGE TRANSFORMS
# -----------------------------

# Training transform (augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224,224)),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor()
])
# Validation transform (clean)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# DATASETS
# -----------------------------

# Training dataset
train_dataset = datasets.ImageFolder(
    root="dataset/train",
    transform=train_transform
)

# Validation dataset
val_dataset = datasets.ImageFolder(
    root="dataset/val",
    transform=val_transform
)

# Print classes
print("Classes:", train_dataset.classes)

# -----------------------------
# DATALOADERS
# -----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

# -----------------------------
# MODEL
# -----------------------------
model = WeldResNet().to(device)

# -----------------------------
# LOSS FUNCTION & OPTIMIZER
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# -----------------------------
# TRAINING
# -----------------------------
epochs = 10

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    # Average epoch loss
    epoch_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{epochs}] Avg Loss: {epoch_loss:.4f}")

print("Training complete!")

# -----------------------------
# VALIDATION
# -----------------------------
model.eval()

correct = 0
total = 0

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # Predicted class
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Accuracy
accuracy = 100 * correct / total

print(f"Validation Accuracy: {accuracy:.2f}%")

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(
    model.state_dict(),
    "weld_model.pth"
)

print("\nModel saved as weld_model.pth")

# -----------------------------
# TEST DATASET
# -----------------------------

test_dataset = datasets.ImageFolder(
    root="dataset/test",
    transform=val_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

# -----------------------------
# TEST EVALUATION
# -----------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total

print(f"\nTest Accuracy: {test_accuracy:.2f}%")

