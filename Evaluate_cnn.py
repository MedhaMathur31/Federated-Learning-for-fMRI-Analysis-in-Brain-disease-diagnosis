import os
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 8 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class FMRIDataset(Dataset):
    def __init__(self, nii_paths, labels):
        self.data = []
        self.labels = labels
        for path in nii_paths:
            img = nib.load(path)
            data = img.get_fdata()
            if data.ndim != 4:
                continue
            data_tensor = torch.tensor(data, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(1)
            resized = F.interpolate(data_tensor, size=(32, 32, 32), mode='trilinear', align_corners=False)
            resized = (resized - resized.mean()) / (resized.std() + 1e-5)
            self.data.append(resized[:10])
        self.data = torch.cat(self.data, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx // 10]

nii_paths = [
    "fmri_data/center1/sub-11/sub-11_bold.nii.gz",
    "fmri_data/center2/sub-12/sub-12_bold.nii.gz",
    "fmri_data/center3/sub-13/sub-13_bold.nii.gz",
]
labels = [0, 1, 1]

dataset = FMRIDataset(nii_paths, labels)
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader = DataLoader(val_set, batch_size=4)

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses, val_losses = [], []

for epoch in range(5):
    model.train()
    epoch_train_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    train_losses.append(epoch_train_loss / len(train_loader))

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item()
    val_losses.append(val_loss / len(val_loader))

# Plot Loss
plt.figure(figsize=(6, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("CNN Training & Validation Loss")
plt.savefig("cnn_train_val_loss.png")
plt.close()

# Confusion Matrix and Report
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for x, y in val_loader:
        preds = torch.argmax(model(x), dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

conf_mat = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["Class 0", "Class 1"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("cnn_confusion_matrix.png")
plt.close()

print("=== Classification Report ===")
print(classification_report(all_labels, all_preds))
