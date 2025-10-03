import os
import pandas as pd
import nibabel as nib
import numpy as np
import shutil
import torch
import torch.nn as nn
import flwr as fl
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Process

# CONFIG
num_sites = 3
subjects_per_site = 20

class VAE(nn.Module):
    def __init__(self, input_dim=32*32*32, latent_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, latent_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 512), nn.ReLU(), nn.Linear(512, input_dim))

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

class LSTM(nn.Module):
    def __init__(self, input_dim=32*32*32, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def load_and_harmonize_data(site_id):
    data, labels = [], []
    for subject in range(subjects_per_site):
        img_path = f"site_{site_id}/site_{site_id}_subject_{subject}.nii.gz"
        label_path = f"site_{site_id}/site_{site_id}_subject_{subject}_label.txt"
        img = nib.load(img_path).get_fdata()
        img = img[:32, :32, :32, :10]
        img = np.resize(img, (10, 32*32*32))
        data.append(img)
        with open(label_path, "r") as f:
            labels.append(int(f.read()))
    data = torch.tensor(np.array(data), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    mean = data.mean(dim=(0, 1), keepdim=True)
    std = data.std(dim=(0, 1), keepdim=True)
    data = (data - mean) / (std + 1e-6)
    return data, labels

class FMRIClient(fl.client.NumPyClient):
    def __init__(self, site_id):
        self.vae = VAE()
        self.cnn = CNN3D()
        self.lstm = LSTM()
        self.data, self.labels = load_and_harmonize_data(site_id)
        self.optimizer_vae = torch.optim.Adam(self.vae.parameters(), lr=0.001)
        self.optimizer_cnn = torch.optim.Adam(self.cnn.parameters(), lr=0.001)
        self.optimizer_lstm = torch.optim.Adam(self.lstm.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in list(self.vae.state_dict().values()) + 
                list(self.cnn.state_dict().values()) + list(self.lstm.state_dict().values())]

    def set_parameters(self, parameters):
        params = [torch.tensor(p) for p in parameters]
        v_len = len(list(self.vae.state_dict().values()))
        c_len = len(list(self.cnn.state_dict().values()))
        l_len = len(list(self.lstm.state_dict().values()))
        self.vae.load_state_dict(dict(zip(self.vae.state_dict().keys(), params[:v_len])))
        self.cnn.load_state_dict(dict(zip(self.cnn.state_dict().keys(), params[v_len:v_len+c_len])))
        self.lstm.load_state_dict(dict(zip(self.lstm.state_dict().keys(), params[-l_len:])))

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.vae.train(); self.cnn.train(); self.lstm.train()
        for _ in range(2):
            self.optimizer_vae.zero_grad()
            recon, mu, log_var = self.vae(self.data[:, 0].reshape(-1, 32*32*32))
            vae_loss = nn.MSELoss()(recon, self.data[:, 0].reshape(-1, 32*32*32)) + \
                       0.01 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            vae_loss.backward(); self.optimizer_vae.step()

            self.optimizer_cnn.zero_grad()
            outputs = self.cnn(self.data[:, 0].reshape(-1, 1, 32, 32, 32))
            cnn_loss = self.criterion(outputs, self.labels)
            cnn_loss.backward(); self.optimizer_cnn.step()

            self.optimizer_lstm.zero_grad()
            lstm_out = self.lstm(self.data.permute(0, 2, 1))
            lstm_loss = self.criterion(lstm_out.squeeze(), self.labels)
            lstm_loss.backward(); self.optimizer_lstm.step()

        return self.get_parameters(config), len(self.data), {}

def start_client(site_id):
    client = FMRIClient(site_id)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

def start_server():
    strategy = fl.server.strategy.FedAvg(min_fit_clients=3, min_available_clients=3)
    fl.server.start_server(server_address="localhost:8080", config=fl.server.ServerConfig(num_rounds=10), strategy=strategy)

if __name__ == "__main__":
    # Dataset already prepared. Skip extract_and_organize_data()

    processes = []
    server_process = Process(target=start_server)
    processes.append(server_process)
    server_process.start()

    for site_id in range(num_sites):
        client_process = Process(target=start_client, args=(site_id,))
        processes.append(client_process)
        client_process.start()

    for p in processes:
        p.join()

    print("Federated learning completed!")
