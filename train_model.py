import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "augmented_colored")

MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "best_dr_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
EPOCHS = 15   # ✅ Final recommended
NUM_CLASSES = 5
VAL_SPLIT = 0.2

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- DATASET ----------------
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print("Classes:", full_dataset.classes)

# ---------------- SAMPLER FIX ----------------
train_targets = [full_dataset.samples[i][1] for i in train_dataset.indices]

class_count = np.bincount(train_targets)
print("Train class counts:", class_count)

class_weights = 1.0 / class_count

sample_weights = [class_weights[label] for label in train_targets]
sample_weights = torch.DoubleTensor(sample_weights)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# ---------------- DATALOADER ----------------
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ----------------
model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, NUM_CLASSES)

model = model.to(DEVICE)

# ---------------- LOSS ----------------
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ---------------- TRAINING ----------------
best_accuracy = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        # ✅ FIX: handle tuple safely
        if isinstance(outputs, tuple):
            main_output, aux_output = outputs
            loss1 = criterion(main_output, labels)
            loss2 = criterion(aux_output, labels)
            loss = loss1 + 0.4 * loss2
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # ---------------- VALIDATION ----------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss:.2f} Accuracy: {accuracy:.2f}%")

    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), MODEL_PATH)
        print("✅ Best model saved!")

print(f"\n🔥 Training Completed! Best Accuracy: {best_accuracy:.2f}%")