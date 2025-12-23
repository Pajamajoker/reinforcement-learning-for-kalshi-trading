"""
Metrics calculation helpers for backtesting and evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict


def calculate_metrics(equity_curve: pd.Series) -> Dict[str, float]:
    """
    Calculate performance metrics from equity curve.

    Args:
        equity_curve: Series of equity values over time.

    Returns:
        Dictionary of metrics: total_return, max_drawdown, sharpe_ratio.
    """
    if len(equity_curve) == 0:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

    initial = equity_curve.iloc[0]
    final = equity_curve.iloc[-1]
    total_return = (final / initial) - 1.0

    # Calculate drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = abs(drawdown.min())

    # Calculate Sharpe ratio (simple version using returns)
    returns = equity_curve.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0.0

    return {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
    }




