import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from rl_environment import MarketMakerEnv

def run_visualizer():
    print("Loading the environment....")
    env=MarketMakerEnv()
    model_path=os.path.join("models","ppo_market_maker_v2_1M")
    try:
        model = PPO.load(model_path)
        print("Successfully loaded V2 Brain!")
    except FileNotFoundError:
        print(f"Error: Could not find {model_path}.zip. Did you save it in the models folder?")
        return
    history_mid = []
    history_bid = []
    history_ask = []
    history_intensity = []
    history_inventory = []
    obs, info = env.reset()
    done = False
    print("Simulating 2000 ticks. Let the AI trade...")
    while not done:
        action,_states = model.predict(obs,deterministic = True)
        current_mid = env.book.get_mid_price() or 100.0
        bid_depth = float(action[0])
        ask_depth = float(action[1])
        my_bid = current_mid - bid_depth
        my_ask = current_mid + ask_depth
        obs, reward, terminated, truncated, info = env.step(action)
        history_mid.append(current_mid)
        history_bid.append(my_bid)
        history_ask.append(my_ask)
        history_intensity.append(env.hawkes.curr_intensity)
        history_inventory.append(env.inventory)
        if terminated or truncated:
            done = True
    print(f"Simulation Finished! Final Inventory: {env.inventory}")
    print(f"Rendering Dashboard...")
    fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize = (12, 10), sharex = True)
    time_steps = np.arange(len(history_mid))

    ax1.plot(time_steps, history_mid, label="Mid Price", color="black", linewidth=1.5)
    ax1.plot(time_steps, history_ask, label="AI Ask Quote", color="red", alpha=0.6, linewidth=1)
    ax1.plot(time_steps, history_bid, label="AI Bid Quote", color="green", alpha=0.6, linewidth=1)
    ax1.set_title("AI Market Maker Quotes vs. Mid Price (1M Steps)")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_steps, history_intensity, color="purple", linewidth=1)
    ax2.fill_between(time_steps, history_intensity, color="purple", alpha=0.2)
    ax2.set_title("Hawkes Process Intensity (Toxic Flow Danger)")
    ax2.set_ylabel("Intensity")
    ax2.grid(True, alpha=0.3)

    ax3.plot(time_steps, history_inventory, color='blue', linewidth=1.5)
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title("Agent Inventory Position")
    ax3.set_xlabel("Time (Ticks)")
    ax3.set_ylabel("Shares Held")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_visualizer()