import pandas as pd
import numpy as np

def calculate_metrics():
    print("Loading Backtest Data...")
    try:
        df = pd.read_csv("data/backtest_results.csv")
    except FileNotFoundError:
        print("Error: Could not find data/backtest_results.csv")
        return
    returns = df["RL_Reward"].values
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    win_rate = np.mean(returns>0)*100

    sharpe_ratio = mean_return/std_return if std_return>0 else 0

    negative_returns=returns[returns<0]
    downside_std=np.std(negative_returns) if len(negative_returns)>0 else 0.000001
    sortino_ratio = mean_return/downside_std

    cumulative_returns = np.cumsum(returns)
    peak=np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - peak
    mdd = np.min(drawdown)

    print("\n" + "="*30)
    print("  INSTITUTIONAL METRICS")
    print("="*30)
    print(f"Total Episodes:   {len(returns)}")
    print(f"Average Profit:   ${mean_return:.2f}")
    print(f"Profit Volatility: ${std_return:.2f}")
    print(f"Win Rate (> $0):  {win_rate:.2f}%")
    print("-" * 30)
    print(f"Sharpe Ratio:     {sharpe_ratio:.3f}")
    print(f"Sortino Ratio:    {sortino_ratio:.3f}")
    print(f"Max Drawdown:     ${mdd:.2f}")
    print("="*30)

if __name__ == "__main__":
    calculate_metrics()