import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Load sub-12 ===
nii_path = "fmri_data/center3/sub-01/sub-01_bold.nii.gz"

img = nib.load(nii_path)
data_np = img.get_fdata()
print("Loaded shape:", data_np.shape)

if data_np.ndim != 4:
    raise ValueError(f"Expected 4D fMRI data but got shape: {data_np.shape}")

# Convert to tensor and rearrange: [T, 1, X, Y, Z]
data_tensor = torch.tensor(data_np, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(1)
# local_test_runner.py

import os
import nibabel as nib
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# ✅ Set correct path to NIfTI file
nii_path = "fmri_data/center3/sub-01/sub-01_bold.nii.gz"

img = nib.load(nii_path)
data = img.get_fdata()
print("Shape of loaded fMRI data:", data.shape)

data_tensor = torch.tensor(data[:32, :32, :32, :10], dtype=torch.float32).permute(3, 0, 1, 2)  # [T, x, y, z]
print("Tensor shape:", data_tensor.shape)

# Models
class VAE(nn.Module):
    def __init__(self, input_dim=32*32*32, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, latent_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 512), nn.ReLU(), nn.Linear(512, input_dim))
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x)
        return h[:, :self.latent_dim], h[:, self.latent_dim:]

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

class CNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(16 * 16 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16 * 16)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Evaluation functions
def plot_confusion_matrix(model, data, true_labels, center_id):
    with torch.no_grad():
        logits = model(data)
        preds = torch.argmax(logits, dim=1)
        cm = confusion_matrix(true_labels.numpy(), preds.numpy())
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - Center {center_id}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"confusion_matrix_center_{center_id}.png")
        plt.close()

def print_classification_report(model, data, true_labels):
    with torch.no_grad():
        logits = model(data)
        preds = torch.argmax(logits, dim=1)
        print("\nClassification Report:\n")
        print(classification_report(true_labels.numpy(), preds.numpy()))

def plot_roc_curve(model, data, true_labels, center_id):
    with torch.no_grad():
        probs = torch.softmax(model(data), dim=1)[:, 1]
        fpr, tpr, _ = roc_curve(true_labels.numpy(), probs.numpy())
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - Center {center_id}")
        plt.legend()
        plt.savefig(f"roc_curve_center_{center_id}.png")
        plt.close()

# === Run test ===
vae = VAE()
cnn = CNN3D()

# Run VAE
data_flat = data_tensor.reshape(10, -1)

recon, mu, log_var = vae(data_flat)
print("VAE reconstruction shape:", recon.shape)

# Run CNN
cnn_input = data_tensor[0].unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 32, 32, 32]
out = cnn(cnn_input)
print("CNN prediction logits:", out)

# Dummy label
label_tensor = torch.tensor([0, 1])  # Example for 2-class test
cnn_input = torch.cat([cnn_input, cnn_input], dim=0)


# Evaluation and visualization
plot_confusion_matrix(cnn, cnn_input, label_tensor, center_id=1)
print_classification_report(cnn, cnn_input, label_tensor)
plot_roc_curve(cnn, cnn_input, label_tensor, center_id=1)

print("✅ Evaluation visualizations saved!")

data_tensor = torch.tensor(data_np[:32, :32, :32, :10], dtype=torch.float32)
data_tensor = data_tensor.permute(3, 0, 1, 2).unsqueeze(1)  # Shape: [10, 1, 32, 32, 32]


# Resize to [T, 1, 32, 32, 32]
resized = torch.nn.functional.interpolate(
    data_tensor[:10], size=(32, 32, 32), mode='trilinear', align_corners=False
)[:10]

# ✅ Normalize before flattening for VAE
resized = (resized - resized.mean()) / (resized.std() + 1e-5)

# ✅ Clamp extreme values
resized = torch.clamp(resized, -3, 3)

# Flatten
data_flat = resized.view(10, -1)



# === Normalize ===
print("Input stats:", resized.min().item(), resized.max().item(), resized.mean().item())
resized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-5)
print("Post-normalization stats:", resized.min().item(), resized.max().item(), resized.mean().item())

# === Define Models ===
class VAE(nn.Module):
    def __init__(self, input_dim=32*32*32, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, latent_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 512), nn.ReLU(), nn.Linear(512, input_dim))
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x)
        return h[:, :self.latent_dim], h[:, self.latent_dim:]

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

vae = VAE()

# === Run VAE ===
recon, mu, log_var = vae(data_flat)
print("Reconstruction stats:", recon[0].min().item(), recon[0].max().item(), recon[0].mean().item())
print("mu stats:", mu.min().item(), mu.max().item(), mu.mean().item())
print("log_var stats:", log_var.min().item(), log_var.max().item(), log_var.mean().item())


# === Visualize ===
recon_3d = recon[0].view(32, 32, 32).detach().numpy()
original_vol = resized[0][0].numpy()

mid_orig = original_vol[:, :, original_vol.shape[2] // 2]
mid_recon = recon_3d[:, :, recon_3d.shape[2] // 2]

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(mid_orig, cmap='gray')
axs[0].set_title("Original Slice")
axs[1].imshow(mid_recon, cmap='gray')
axs[1].set_title("Reconstructed Slice")
plt.suptitle("VAE Reconstruction - Center 2 (sub-12)")
plt.tight_layout()
plt.savefig("vae_reconstruction_center2.png")
plt.show()
