# train_loop.py
import numpy as np
import pandas as pd
import torch

# Import your helpers
from actor_critic import Actor
from ml_dl_models.actor_critic import Critic
from ml_dl_models.rnn_vae import RNNVAEEncoder
from structural_breaks.bocpd import BOCPD
from util.running_mean_std import RunningMeanStd
from util.weighted_replay_buffer import WeightedReplayBuffer


def train_loop(price_stream, state_window=50, seq_len_for_vae=50,
               total_steps=5000, bocpd_hazard=300.0, device="cpu"):
    """
    Main training loop using spread series as input.
    price_stream: pandas Series (index = dates, values = spread) or numpy array.
    """
    from util.running_mean_std import RunningMeanStd
    # --- Convert input ---
    if isinstance(price_stream, pd.Series):
        prices = price_stream.values
    else:
        prices = np.asarray(price_stream)

    # --- Initialize components ---
    rms = RunningMeanStd()
    actor = Actor(state_dim=state_window, latent_dim=8).to(device)
    critic = Critic(state_dim=state_window, latent_dim=8).to(device)
    encoder = Encoder().to(device)
    bocpd = BOCPD(hazard=bocpd_hazard)

    # TODO: your training logic here

    return actor, critic, encoder, bocpd


# Optional: quick test
if __name__ == "__main__":
    dummy_series = pd.Series(np.random.randn(2000),
                             index=pd.date_range("2020-01-01", periods=2000))
    actor, critic, encoder, bocpd = train_loop(dummy_series)
    print("train_loop ran successfully!")
