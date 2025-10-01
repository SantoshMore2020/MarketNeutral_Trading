# train_loop.py
import math
import numpy as np
import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import pandas as pd

# Import your helpers
# from ml_dl_models.actor_critic import Actor
# from ml_dl_models.actor_critic import Critic
# from ml_dl_models.rnn_vae import RNNVAEEncoder
# from structural_break.bocpd import BOCPD
# from util.running_mean_std import RunningMeanStd
# from util.weighted_replay_buffer import WeightedReplayBuffer


# def train_loop(price_stream, state_window=50, seq_len_for_vae=50,
#                total_steps=5000, bocpd_hazard=300.0, device="cpu"):
#     """
#     Main training loop using spread series as input.
#     price_stream: pandas Series (index = dates, values = spread) or numpy array.
#     """
#     from util.running_mean_std import RunningMeanStd
#     from structural_break.bocpd import BOCPD
#     from ml_dl_models.rnn_vae import RNNVAEEncoder
#     from ml_dl_models.actor_critic import Actor
#     from ml_dl_models.actor_critic import Critic
    
    
#     # --- Convert input ---
#     if isinstance(price_stream, pd.Series):
#         prices = price_stream.values
#     else:
#         prices = np.asarray(price_stream)

#     # --- Initialize components ---
#     rms = RunningMeanStd()
#     actor = Actor(state_dim=state_window, latent_dim=8).to(device)
#     critic = Critic(state_dim=state_window, latent_dim=8).to(device)
#     encoder = RNNVAEEncoder().to(device)
#     bocpd = BOCPD(hazard=bocpd_hazard)

#     # TODO: your training logic here

#     return actor, critic, encoder, bocpd


# # Optional: quick test
# if __name__ == "__main__":
#     dummy_series = pd.Series(np.random.randn(2000),
#                              index=pd.date_range("2020-01-01", periods=2000))
#     actor, critic, encoder, bocpd = train_loop(dummy_series)
#     print("train_loop ran successfully!")
def train_loop(
    price_stream,
    state_window=50,
    seq_len_for_vae=50,
    total_steps=10000,
    bocpd_hazard=300.0,
    device='cpu'
):
    if isinstance(price_stream, pd.Series):
        prices = price_stream.values  # just the spread values
        dates = price_stream.index    # keep dates for later if you want plotting
    else:
        prices = np.asarray(price_stream)
        dates = None

    from util.running_mean_std import RunningMeanStd
    from structural_break.bocpd import BOCPD
    from ml_dl_models.rnn_vae import RNNVAEEncoder
    from ml_dl_models.actor_critic import Actor
    from ml_dl_models.actor_critic import Critic
    
    # state: recent normalized returns (window)
    rms = RunningMeanStd()
    bocpd = BOCPD(hazard_lambda=bocpd_hazard)
    input_dim = 2  # [return, bocpd_prob] per timestep into encoder
    z_dim = 8
    state_dim = state_window  # using flattened returns as state; in practice use richer features
    actor = Actor(state_dim=state_dim, z_dim=z_dim).to(device)
    critic = Critic(state_dim=state_dim, z_dim=z_dim).to(device)
    encoder = RNNVAEEncoder(input_dim=input_dim, hidden_dim=64, z_dim=z_dim).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-4)
    enc_opt = optim.Adam(encoder.parameters(), lr=3e-4)

    buffer = WeightedReplayBuffer(capacity=30000)

    # action noise base sigma
    base_action_sigma = 0.05

    # walkthrough
    # prices = price_stream
    # returns = np.log(prices[1:] / prices[:-1])
    spread = prices
    returns = np.diff(spread)

    T = min(total_steps, len(returns) - state_window - 1)
    # initial running mean/std update
    #rms.update(returns[:1000])
    rms.update(returns[:100])

    # initialize state: last `state_window` returns
    idx = 0
    state_returns = list(returns[idx: idx + state_window])
    idx += state_window
    last_action = 0.0

    for step in trange(T):
        cur_ret = returns[idx]
        rms.update([cur_ret])
        # BOCPD expects scalar observation -> use normalized return
        norm_ret = float((cur_ret - rms.mean) / (math.sqrt(rms.var) + 1e-8))
        change_prob = bocpd.update(norm_ret)  # float in [0,1]
        # build encoder input sequence (seq_len_for_vae)
        seq_start = max(0, idx - seq_len_for_vae + 1)
        seq_rets = returns[seq_start: idx + 1]
        # pad if needed
        if len(seq_rets) < seq_len_for_vae:
            pad = np.zeros(seq_len_for_vae - len(seq_rets))
            seq_rets = np.concatenate([pad, seq_rets])
        # form encoder input: (seq_len, input_dim) where input_dim = [norm_ret, change_prob]
        seq_inp = np.stack([ (seq_rets - rms.mean) / (math.sqrt(rms.var)+1e-8),
                              np.ones_like(seq_rets) * change_prob ], axis=-1)[None, ...]  # batch=1
        seq_inp_t = torch.tensor(seq_inp, dtype=torch.float32).to(device)
        z_t, mu, logvar = encoder(seq_inp_t)  # z_t shape (1, z_dim)
        # state vector for policy: flatten last `state_window` normalized returns
        state_arr = np.array(state_returns[-state_window:])
        state_norm = (state_arr - rms.mean) / (math.sqrt(rms.var) + 1e-8)

        state_t = torch.tensor(state_norm.astype(np.float32))[None, :].to(device)
        with torch.no_grad():
            z_t_det = z_t  # already batch=1
            action_mean = actor(state_t, z_t_det).cpu().numpy().squeeze()
        # exploration scale increases with change_prob
        noise_sigma = base_action_sigma * (1.0 + 5.0 * change_prob)  # alpha=5 scaling
        action = action_mean + np.random.normal(scale=noise_sigma, size=action_mean.shape)
        action = np.clip(action, -1.0, 1.0)
        # interpret action: e.g., fraction of capital to long (positive) or short (negative)
        # reward: simple PnL = action * next_return
        next_ret = returns[idx + 1]
        reward = float(action * next_ret)  # simple linear P&L

        # store transition in buffer with initial weight 1.0
        buffer.push(
            state_norm.astype(np.float32),
            action.astype(np.float32),
            reward,
            None, False,
            1.0,
            seq_inp.squeeze(0).astype(np.float32)  # shape (seq_len, input_dim)
        )

        # if change_prob large, upweight recent transitions
        if change_prob > 0.15:
            buffer.upweight_recent(window=200, multiplier=1.8)

        # periodic updates
        if buffer.size() >= 256 and step % 16 == 0:
            batch = buffer.sample(128)

            # prepare tensors for actor/critic
            states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32).to(device)
            actions = torch.tensor(np.stack([b.action for b in batch]), dtype=torch.float32).to(device)
            rewards = torch.tensor(np.stack([b.reward for b in batch]), dtype=torch.float32).unsqueeze(-1).to(device)

            # critic update
            values = critic(states, torch.zeros(states.size(0), z_dim).to(device))  # critic input placeholder z
            targets = rewards
            critic_loss = nn.MSELoss()(values, targets)
            critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

            # actor update
            with torch.no_grad():
                adv = (rewards - values).detach()
            pred_actions = actor(states, torch.zeros(states.size(0), z_dim).to(device))
            actor_loss = - (pred_actions * adv).mean()
            actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

            # encoder update (proper VAE training)
            seqs = torch.tensor(np.stack([b.seq_inp for b in batch]), dtype=torch.float32).to(device)
            z_batch, mu, logvar = encoder(seqs)

            # KL divergence for VAE
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

            enc_loss = 0.001 * kl
            enc_opt.zero_grad()
            enc_loss.backward()
            enc_opt.step()

        # slide window
        state_returns.append(next_ret)
        idx += 1

    print("Training loop finished. Buffer size:", buffer.size())
    return actor, critic, encoder, bocpd
