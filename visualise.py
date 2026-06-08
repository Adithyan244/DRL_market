import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_profit_distribution():
    print("Loading Backtest Data...")
    try:
        df = pd.read_csv("data/backtest_results.csv")
    except FileNotFoundError:
        print("Error: Could not find data/backtest_results.csv.")
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    sns.kdeplot(df['RL_Reward'], fill=True, color="dodgerblue", alpha=0.5, linewidth=2, label="AI Profit Distribution")

    rl_mean = df['RL_Reward'].mean()
    as_baseline = 12.80 

    plt.axvline(rl_mean, color='blue', linestyle='--', linewidth=2, label=f'AI Average: ${rl_mean:.2f}')
    plt.axvline(as_baseline, color='red', linestyle='--', linewidth=2, label=f'AS Baseline: ${as_baseline:.2f}')

    plt.title("Profit Distribution: RL Agent vs. Mathematical Baseline (1,000 Trading Days)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Profit per Episode ($)", fontsize=12)
    plt.ylabel("Density (Frequency)", fontsize=12)
    plt.legend(loc='upper left', fontsize=11)

    plt.tight_layout()
    print("Rendering KDE Chart...")
    plt.show()

if __name__ == "__main__":
    plot_profit_distribution()