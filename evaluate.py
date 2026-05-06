import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "augmented_colored")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_dr_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
NUM_CLASSES = 5

labels = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- DATASET ----------------
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ----------------
model = models.inception_v3(weights=None, aux_logits=True)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully")

# ---------------- PREDICTIONS ----------------
y_true = []
y_pred = []

with torch.no_grad():
    for images, targets in loader:
        images = images.to(DEVICE)

        outputs = model(images)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        _, predicted = torch.max(outputs, 1)

        y_true.extend(targets.numpy())
        y_pred.extend(predicted.cpu().numpy())

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)

# ---------------- PRINT REPORT ----------------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels))

# ---------------- PLOT ----------------
plt.figure(figsize=(12, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Confusion Matrix: Diabetic Retinopathy Classification", fontsize=20)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

print("\n✅ Confusion matrix saved as confusion_matrix.png")