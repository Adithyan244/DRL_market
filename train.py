import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from rl_environment import MarketMakerEnv

print("Loading Market Maker Environment....")
env=make_vec_env(lambda:MarketMakerEnv(),n_envs=1)
print("Initialising PPO Agent....")
model=PPO("MlpPolicy",env,verbose=1,learning_rate=0.0003,gamma=0.99,tensorboard_log="./ppo_tensorboard/")
print("Start thr training loop... Let the AI get to Work....")
model.learn(total_timesteps=1000000)
save_dir="models"
os.makedirs(save_dir,exist_ok=True)
save_path=os.path.join(save_dir,"ppo_market_maker_v2_1M")
model.save(save_path)
print(f"Training Complete! Neural Network weights saved to {save_path}.zip")
