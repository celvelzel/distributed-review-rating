"""
Post-processing: optimal rounding + clipping for rating predictions.
"""
import numpy as np


def optimal_round(predictions, y_true, granularity=0.5):
    """
    Find the best rounding strategy by grid-searching thresholds on OOF data.

    Strategies compared:
      - round to nearest integer (standard round)
      - floor (np.floor + 1 if needed, but ratings are 1-5 so floor works)
      - ceil (np.ceil)
      - round to nearest `granularity` step (0.5 steps)

    Returns:
        best_preds: optimally rounded predictions
        best_name: name of the best strategy
        best_rmse: RMSE of the best strategy
    """
    def rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    # Integer thresholds for boundary rounding
    thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]

    strategies = {}

    # Strategy 1: standard round to integer
    rounded = np.round(predictions).astype(np.float32)
    strategies["round"] = rounded

    # Strategy 2: floor to integer
    floored = np.floor(predictions).astype(np.float32)
    strategies["floor"] = floored

    # Strategy 3: ceil to integer
    ceiled = np.ceil(predictions).astype(np.float32)
    strategies["ceil"] = ceiled

    # Strategy 4: round to nearest 0.5
    rounded_half = np.round(predictions / granularity) * granularity
    strategies["round_half"] = rounded_half.astype(np.float32)

    # Strategy 5: round with shifted thresholds
    for t in thresholds:
        shifted = np.where(predictions >= t, np.ceil(predictions), np.floor(predictions))
        shifted = shifted.astype(np.float32)
        strategies[f"shifted_{t}"] = shifted

    # Find best strategy
    best_name = None
    best_rmse = float("inf")
    best_preds = None

    for name, preds in strategies.items():
        score = rmse(preds, y_true)
        if score < best_rmse:
            best_rmse = score
            best_name = name
            best_preds = preds

    return best_preds, best_name, best_rmse


def clip_15(predictions):
    """Clip predictions to [1, 5] rating range."""
    return np.clip(predictions, 1.0, 5.0).astype(np.float32)
