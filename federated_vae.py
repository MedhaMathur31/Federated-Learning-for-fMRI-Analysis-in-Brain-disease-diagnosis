import os
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# === VAE Model ===
class VAE(nn.Module):
    def __init__(self, input_dim=32*32*32, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(), nn.Linear(512, input_dim)
        )
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

# === Utilities ===
def load_fmri_tensor(nii_path):
    data = nib.load(nii_path).get_fdata()
    if data.ndim != 4:
        raise ValueError(f"Expected 4D fMRI data but got shape: {data.shape}")
    tensor = torch.tensor(data, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(1)
    resized = F.interpolate(tensor, size=(32, 32, 32), mode='trilinear', align_corners=False)[:10]
    resized = (resized - resized.mean()) / (resized.std() + 1e-5)
    resized = torch.clamp(resized, -3, 3)
    return resized.view(10, -1)

def vae_loss(recon, x, mu, log_var):
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_div

# === Setup ===
subjects = {
    "center1": "fmri_data/center1/sub-11/sub-11_bold.nii.gz",
    "center2": "fmri_data/center2/sub-12/sub-12_bold.nii.gz",
    "center3": "fmri_data/center3/sub-13/sub-13_bold.nii.gz",
}

vae = VAE()
rounds = 3
lr = 1e-3
optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

train_losses = []

# === Federated Training Loop ===
for rnd in range(rounds):
    print(f"\n=== Federated Round {rnd + 1} ===")
    local_losses = []

    for center, nii_path in subjects.items():
        print(f"\n[{center}] -> Loading {nii_path}")
        data = load_fmri_tensor(nii_path)
        optimizer.zero_grad()
        recon, mu, log_var = vae(data)
        loss = vae_loss(recon, data, mu, log_var)
        loss.backward()
        optimizer.step()
        local_losses.append(loss.item())
        print(f"[{center}] Loss: {loss.item():.2f}")

    avg_loss = sum(local_losses) / len(local_losses)
    train_losses.append(avg_loss)
    print(f"Round {rnd + 1} Average Loss: {avg_loss:.2f}")

# === Plotting Loss Curve ===
plt.figure(figsize=(6, 4))
plt.plot(range(1, rounds + 1), train_losses, marker='o')
plt.title("Federated VAE Training Loss Across Rounds")
plt.xlabel("Federated Round")
plt.ylabel("Average Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("federated_training_loss.png")
print("\n✅ Saved federated training loss curve as 'federated_training_loss.png'")
