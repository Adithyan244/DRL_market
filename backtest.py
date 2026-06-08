import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from stable_baselines3 import PPO
from rl_environment import MarketMakerEnv

def run_monte_carlo_backtest(episodes=1000):
    print(f"Loading 6-Variable Environment and V2 Brain...")
    env = MarketMakerEnv()
    model_path = os.path.join("models", "ppo_market_maker_v2_1M")
    try:
        model=PPO.load(model_path)
    except FileNotFoundError:
        print(f"Error: Could not find {model_path}.zip")
        return
    rl_pnl_results = []
    rl_inventory_results = []
    print(f"Running Model Carlo Simulation ({episodes} Episodes)...")
    for i in tqdm(range(episodes)):
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                done = True
        final_mid = env.book.get_mid_price() or 100.0
        final_pnl = env.balance + (env.inventory * final_mid)
        inventory_penalty = 0.01*(env.inventory**2)
        final_reward = final_pnl - inventory_penalty
        rl_pnl_results.append(final_reward)
        rl_inventory_results.append(env.inventory)
    
    df=pd.DataFrame({
        "RL_Reward" : rl_pnl_results,
        "RL_Final_Inventory": rl_inventory_results
    })
    os.makedirs("data",exist_ok=True)
    csv_path="data/backtest_results.csv"
    df.to_csv(csv_path, index = False)
    print("\n--- BackTest Complete ---")
    print(f"Average RL Reward: ${np.mean(rl_pnl_results):.2f}")
    # AS Baseline scored $12.80
    win_rate = np.mean(np.array(rl_pnl_results)>12.80)*100
    print(f"RL Win Rate vs. AS Baseline ($12.80): {win_rate:.2f}%")
    print(f"Data Successfully saved to {csv_path}")

if __name__ == "__main__":
    run_monte_carlo_backtest(episodes=1000)