import torch
import torch.nn as nn

# -------------------------
# RNN-VAE latent encoder
# -------------------------
#class RNNVAEEncoder(nn.Module):
class RNNVAEEncoder:
    def __init__(self, input_dim, hidden_dim=64, z_dim=8, n_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x_seq):
        # x_seq: (batch, seq_len, input_dim)
        out, h = self.gru(x_seq)  # out: (batch, seq_len, hidden)
        last = out[:, -1, :]
        mu = self.fc_mu(last)
        logvar = self.fc_logvar(last)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar
