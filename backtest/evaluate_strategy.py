import numpy as np
import matplotlib.pyplot as plt
import torch
from util.running_mean_std import RunningMeanStd

def evaluate_strategy(actor, encoder, price_stream, state_window=50, seq_len_for_vae=50):
    if isinstance(price_stream, pd.Series):
        prices = price_stream.values  # just the spread values
        dates = price_stream.index    # keep dates for later if you want plotting
    else:
        prices = np.asarray(price_stream)
        dates = None

    rms = RunningMeanStd()
    spread = prices  # your series of spreads
    returns = np.diff(spread)  # spread changes
    rms.update(returns[:100])  # warm-up stats

    pnl = []
    capital = 1.0
    state_returns = list(returns[:state_window])

    for t in range(state_window, len(returns)-1):
        # state normalization
        state_arr = np.array(state_returns[-state_window:])
        state_norm = (state_arr - rms.mean) / (np.sqrt(rms.var) + 1e-8)
        state_t = torch.tensor(state_norm.astype(np.float32))[None, :]

        # sequence input for encoder
        seq_rets = returns[t-seq_len_for_vae+1:t+1] if t >= seq_len_for_vae else returns[:t+1]
        if len(seq_rets) < seq_len_for_vae:
            seq_rets = np.concatenate([np.zeros(seq_len_for_vae - len(seq_rets)), seq_rets])
        seq_inp = np.stack([
            (seq_rets - rms.mean) / (np.sqrt(rms.var)+1e-8),
            np.zeros_like(seq_rets)
        ], axis=-1)[None, ...]  # shape (1, seq_len, 2)

        with torch.no_grad():
            z_t, _, _ = encoder(torch.tensor(seq_inp, dtype=torch.float32))
            action = actor(state_t, z_t).numpy().squeeze()

        # apply action to next return
        next_ret = returns[t+1]
        reward = action * next_ret
        capital *= (1 + reward)
        pnl.append(capital)

        # update rolling state
        state_returns.append(next_ret)
        rms.update([returns[t]])

    pnl = np.array(pnl)

    # === Plot ===
    plt.figure(figsize=(16,12))
    # plt.plot(pnl, label="Equity Curve") # Comment out the original plot line
    plt.plot_date(dates[state_window+2: len(dates)], pnl, linestyle='-', marker='None', label="Equity Curve") # Use plot_date with dates and pnl
    plt.axhline(y=1, color='r', linestyle='--', label="Buy and Hold Strategy")
    # if dates is not None:
    #     plt.xticks(range(len(dates)), dates, rotation=90)
    plt.title("Equity Curve of Trained Strategy")
    plt.xlabel("Time")
    plt.ylabel("Capital")
    plt.legend()
    plt.grid(True)
    plt.show()

    # === Metrics ===
    final_capital = pnl[-1]
    sharpe = np.mean(np.diff(pnl)) / (np.std(np.diff(pnl)) + 1e-8) * np.sqrt(252)
    drawdown = np.max(np.maximum.accumulate(pnl) - pnl)

    print(f"Final Capital: {final_capital:.3f}")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Max Drawdown: {drawdown:.3f}")

    return pnl, {"final_capital": final_capital, "sharpe": sharpe, "drawdown": drawdown}
