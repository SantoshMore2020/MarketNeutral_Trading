import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Simple Baseline: Z-Score Pairs Trading
# ============================================
def zscore_trading(spread, window=50, entry=2.0, exit=0.5):
    #spread = df["A"] - df["B"]
    mu = spread.rolling(window).mean()
    sigma = spread.rolling(window).std()
    zscore = (spread - mu) / sigma
    position = np.where(zscore > entry, -1,
                np.where(zscore < -entry, 1,
                np.where(abs(zscore) < exit, 0, np.nan)))
    position = pd.Series(position).ffill().fillna(0)

    pos_shift = position.shift(1)
    pos_shift.iloc[0] = 0
    sprd_diff = spread.diff()
    sprd_diff.iloc[0] = 0
    pnl = pd.Series(pos_shift.values * sprd_diff.values, index = sprd_diff.index)
    pnl.index = pd.to_datetime(pnl.index)
    pnl.plot(title="Baseline PnL (Z-Score Pairs Trading)")
    plt.show()
    pnl = pnl.dropna()
    pnlcum = pnl.cumsum()
    pnlcum.plot(title="Cumulative PnL (Z-Score Pairs Trading)")
    pnlcum.index = pd.to_datetime(pnlcum.index)
    plt.show()
    return pnlcum, zscore, position
