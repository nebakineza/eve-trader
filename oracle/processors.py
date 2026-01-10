import numpy as np
import pandas as pd
from functools import lru_cache
from oracle.config_loader import GLOBAL_CONFIG

@lru_cache(maxsize=16)
def get_weights_ffd(d, thres=1e-5):
    """
    Fixed Window Fractional Differentiation weights.
    d: coefficient (e.g., 0.4)
    thres: cutoff for weight significance
    Cached for performance.
    """
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d, thres=1e-5):
    """
    Apply Fixed Window Fractional Differentiation to a pandas Series.
    Vectorized implementation using np.convolve for latency < 2ms.
    """
    # 1. Compute weights (Cached)
    w = get_weights_ffd(d, thres)
    # w is shape (width, 1), flatten for convolve
    w_flat = w.flatten() 
    width = len(w_flat) - 1
    
    # 2. Prepare Data
    series_clean = series.fillna(method='ffill')
    vals = series_clean.values
    
    if len(vals) < width + 1:
        return pd.Series(np.zeros(len(series)), index=series.index)
        
    # 3. Vectorized Convolution
    # w_flat is [w_k, w_{k-1}, ..., w_0]. 
    # np.convolve(a, v, mode='valid') computes the dot product sliding window.
    # Because w is already reversed (w[::-1] in get_weights), it aligns with the time window
    # [x_{t-k}, ..., x_t] correctly for a direct dot product if we treat it as cross-correlation.
    # However, np.convolve flips the second array (kernel).
    # If w_flat is [w_k, ... w_0], flipping it gives [w_0, ... w_k].
    # Convolution: (f * g)[n] = sum f[m] g[n-m].
    # We want: y[t] = w_0*x[t] + w_1*x[t-1] + ... + w_k*x[t-k].
    # This corresponds to convolving x with [w_0, w_1, ..., w_k].
    # get_weights_ffd returns [w_k, ..., w_1, w_0].
    # So if we simply pass w_flat to convolve, convolve will flip it to [w_0, ..., w_k]
    # and compute exactly what we want.
    
    res_valid = np.convolve(vals, w_flat, mode='valid')
    
    # 4. Padding
    # 'valid' returns N - width + 1 points. We need to pad the beginning with NaNs
    # to maintain shape matching the original index.
    padding = np.full(width, np.nan)
    res_full = np.concatenate([padding, res_valid])
    
    # Ensure index alignment
    if len(res_full) != len(series):
        # Fallback if dimensions mismatch slightly due to valid mode edge cases
        # Should not happen with correct padding
        return pd.Series(np.zeros(len(series)), index=series.index)

    return pd.Series(res_full, index=series.index)

def apply_stationarity(df, d=None):
    """
    Applies FracDiff to Price and Volume columns.
    """
    # Load d from config if available, else use default arg
    if d is None:
        d = GLOBAL_CONFIG.get("model", {}).get("d_coefficient", 0.4)
    
    # Ensure d is float
    d = float(d)

    if 'close' in df.columns:
        df['frac_close'] = frac_diff_ffd(df['close'], d=d)
        # Forward fill the initial NaNs created by the window
        df['frac_close'] = df['frac_close'].fillna(method='bfill')
        
    if 'volume' in df.columns:
        # Volume is often stationary-ish, but FracDiff can help smooth spikes
        # Only log-transform if not already done, usually volume is raw here
        vol_series = df['log_volume'] if 'log_volume' in df.columns else np.log1p(df['volume'])
        df['frac_volume'] = frac_diff_ffd(vol_series, d=d)
        df['frac_volume'] = df['frac_volume'].fillna(method='bfill')
        
    return df
