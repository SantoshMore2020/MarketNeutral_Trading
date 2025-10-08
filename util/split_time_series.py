import numpy as np

def split_time_series(data, train_ratio=0.7, val_ratio=0.15):
    """
    Split data into train, validation, and test sets.
    Assumes data is a pandas series.
    """
    
    n = len(data)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data
