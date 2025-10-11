import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange
import math
import random
from collections import deque, namedtuple
import pandas as pd
from util.running_mean_std import RunningMeanStd
from structural_break.bocpd import BOCPD
from structural_break.hazard import ConstantHazard
from structural_break.distribution import StudentT
from ml_dl_models.rnn_vae import VAEEncoder, vae_loss
from ml_dl_models.lstm import LSTMPolicy
from util.weighted_replay_buffer import WeightedReplayBuffer


def train_loop_lstm(
    stream,
    num_epochs=50,
    save_dir="checkpoints_lstm",
    state_window=50,
    seq_len_for_vae=50,
    total_steps=10000,
    bocpd_hazard=300.0,
    device='cpu'
):
    # ----- Data setup -----
    if isinstance(stream, pd.Series):
        data = stream.values
    else:
        data = np.asarray(stream)

    bocpd = BOCPD(ConstantHazard(bocpd_hazard), StudentT(mu=0, kappa=1, alpha=1, beta=1))
    input_dim = 2
    z_dim = 16
    state_dim = state_window

    encoder = VAEEncoder(input_dim=input_dim, hidden_dim=128, z_dim=z_dim, seq_len=seq_len_for_vae).to(device)
    policy_lstm = LSTMPolicy(input_dim=state_dim + z_dim, hidden_dim=128).to(device)
    #policy_actor = Actor(state_dim=state_dim, z_dim=z_dim, action_dim=1).to(device)


    opt_vae = optim.Adam(encoder.parameters(), lr=1e-3)
    opt_policy = optim.Adam(policy_lstm.parameters(), lr=1e-4)
    # opt_policy = optim.Adam(policy_actor.parameters(), lr=1e-4)


    buffer = WeightedReplayBuffer(capacity=30000)

    # ----- Training loop -----
    for epoch in range(num_epochs):
        rms = RunningMeanStd()
        base_action_sigma = 0.05
        T = min(total_steps, len(data) - state_window - 1)

        rms.update(data[:state_window])
        idx = 0
        state_returns = list(data[idx: idx + state_window])
        idx += state_window

        total_recon, total_kl, total_policy_loss = 0, 0, 0

        for step in trange(T):
            cur_ret = data[idx]
            rms.update([cur_ret])
            norm_ret = float((cur_ret - rms.mean) / (math.sqrt(rms.var) + 1e-8))

            # --- BOCPD change-point probability ---
            change_prob = bocpd.update(norm_ret)

            # --- Encoder (VAE) ---
            seq_start = max(0, idx - seq_len_for_vae + 1)
            seq_rets = data[seq_start: idx + 1]
            if len(seq_rets) < seq_len_for_vae:
                pad = np.zeros(seq_len_for_vae - len(seq_rets))
                seq_rets = np.concatenate([pad, seq_rets])

            seq_inp = np.stack([
                (seq_rets - rms.mean) / (math.sqrt(rms.var) + 1e-8),
                np.ones_like(seq_rets) * change_prob
            ], axis=-1)[None, ...]

            seq_inp_t = torch.tensor(seq_inp, dtype=torch.float32).to(device)
            x_hat, mu, logvar, z_t = encoder(seq_inp_t)
            kl_w = min(1.0, epoch / 100)
            loss_vae, recon_loss, kl_loss = vae_loss(seq_inp_t, x_hat, mu, logvar, kl_weight=kl_w)
            opt_vae.zero_grad(); loss_vae.backward(); opt_vae.step()

            # --- Policy (Actor) ---
            state_arr = np.array(state_returns[-state_window:])
            state_norm = (state_arr - rms.mean) / (math.sqrt(rms.var) + 1e-8)
            state_t = torch.tensor(state_norm.astype(np.float32))[None, :].to(device)

            inp_t = torch.cat([state_t, z_t.detach()], dim=-1)
            action_t = policy_lstm(inp_t)
            # action_t = policy_actor(state_t, z_t.detach())

            action_t = torch.tanh(action_t)  # [-1, 1]
            next_ret = data[idx + 1]
            pnl = action_t * next_ret

            # --- Policy loss (maximize pnl, smooth actions) ---
            loss_policy = -pnl.mean()  # maximize profit
            opt_policy.zero_grad(); loss_policy.backward(); opt_policy.step()

            total_policy_loss += loss_policy.item()
            total_recon += recon_loss
            total_kl += kl_loss

            # --- Store transition in buffer (for future stability) ---
            buffer.push(state_norm.astype(np.float32),
                        action_t.detach().cpu().numpy().astype(np.float32),
                        pnl.item(),
                        None, False, 1.0,
                        seq_inp.squeeze(0).astype(np.float32))

            # Upweight near detected changes
            if change_prob > 0.5:
                buffer.upweight_recent(window=200, multiplier=1.8)

            # Move window
            state_returns.append(next_ret)
            idx += 1

        print(f"Epoch {epoch:03d} | recon={total_recon/len(data):.4f} | kl={total_kl/len(data):.4f} | policy={total_policy_loss/len(data):.4f}")

    print("LSTM policy training complete.")

# # Optional: quick test
# if __name__ == "__main__":
#     dummy_series = pd.Series(np.random.randn(2000),
#                              index=pd.date_range("2020-01-01", periods=2000))
#     train_loop_lstm(dummy_series)
#     print("train_loop ran successfully!")
