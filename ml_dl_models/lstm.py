import math
import numpy as np
import torch
import torch.nn as nn

# -------------------------
# LSTM model
# -------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)
    def forward(self, x):   # x: (batch, seq_len, input_dim)
        x = torch.nan_to_num(x)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)   # return prediction (batch,)
