import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from lob_simulator import limitOrderBook, Hawkes_Process

class MarketMakerEnv(gym.Env):
    def __init__(self):
        super(MarketMakerEnv,self).__init__()
        self.action_space = spaces.Box(
            low=np.array([0.01,0.01]),
            high=np.array([2.00,2.00]),
            dtype=np.float32
        )
        obs_low=np.array([-500.0,0.0,0.0,0.0,-100.0,0.0])
        obs_high=np.array([500.0,np.inf,10.0,1.0,100.0,np.inf])
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )
        self.book = None
        self.hawkes = None
        self.inventory = 0
        self.balance = 0.0
        self.current_step = 0
        self.max_steps = 2000
        self.mid_history=[]
        self.window_size=20

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.book=limitOrderBook()
        self.hawkes=Hawkes_Process(mu=0.05,alpha=0.10,beta=0.15)
        self.book.add_limit_order('bid',99.5,100)
        self.book.add_limit_order('ask',100.5,100)
        self.inventory=0
        self.balance=0.0
        self.current_step=0
        initial_mid=self.book.get_mid_price() or 100.0
        self.mid_history=[initial_mid]*self.window_size
        initial_spread=self.book.get_best_ask()-self.book.get_best_bid()
        initial_intensity=self.hawkes.curr_intensity
        initial_ofi = 0.0
        initial_vol = 0.0
        obs=np.array([
            self.inventory,
            initial_mid,
            initial_spread,
            initial_intensity,
            initial_ofi,
            initial_vol
        ], dtype=np.float32)
        info={}
        return obs,info
    
    def step(self,action):
        self.current_step+=1
        bid_depth=float(action[0])
        ask_depth=float(action[1])
        current_mid=self.book.get_mid_price()
        if current_mid is None:
            current_mid=100.0
        my_bid_price=round(current_mid-bid_depth,2)
        my_ask_price=round(current_mid+ask_depth,2)
        previous_portfolio_value=self.balance+(self.inventory*current_mid)
        market_order_arrived=self.hawkes.generate_order()
        self.hawkes.step(event_occured=market_order_arrived)
        step_ofi=0.0
        if market_order_arrived:
            side=random.choice(['buy','sell'])
            volume=random.randint(10,50)
            if side == 'buy':
                step_ofi = float(volume)
            else:
                step_ofi = -float(volume)
            best_bid=self.book.get_best_bid() or 99.50
            best_ask=self.book.get_best_ask() or 100.50
            if side == 'sell' and my_bid_price>=best_bid:
                self.inventory+=1
                self.balance-=my_bid_price
            elif side == 'buy' and my_ask_price<=best_ask:
                self.inventory-=1
                self.balance+=my_ask_price
            self.book.execute_market_order(side,volume)
        else:
            if random.random()<0.60:
                maker_side = random.choice(['bid','ask'])
                maker_vol = random.randint(10,30)
                if maker_side == 'bid':
                    p=round(current_mid-bid_depth,2)
                else:
                    p=round(current_mid+ask_depth,2)
                self.book.add_limit_order(maker_side,p,maker_vol)
        new_mid=self.book.get_mid_price()
        if new_mid is None:
            new_mid=100.0
        current_portfolio_value=self.balance+(self.inventory*new_mid)
        step_pnl=current_portfolio_value-previous_portfolio_value
        reward=step_pnl-0.01*(self.inventory**2)
        best_bid=self.book.get_best_bid()
        best_ask=self.book.get_best_ask()
        if best_bid is None or best_ask is None:
            new_spread=4.0
        else:
            new_spread=best_ask-best_bid
        self.mid_history.append(new_mid)
        if len(self.mid_history) > self.window_size:
            self.mid_history.pop(0)
        trailing_vol=float(np.std(self.mid_history))
        obs=np.array([
            self.inventory,
            new_mid,
            new_spread,
            self.hawkes.curr_intensity,
            step_ofi,
            trailing_vol
        ],dtype=np.float32)
        terminated=False
        truncated=False
        if self.current_step>=self.max_steps:
            truncated=True
        if self.inventory>=100 or self.inventory<=-100:
            terminated=True
            reward-=1000
        info={}
        return obs,reward,terminated,truncated,info
    

if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    print("Instantiating the Market Maker Environment...")
    env = MarketMakerEnv()
    print("Running the Stable-Baselines3 Environment Checker...")
    check_env(env, warn=True)
    print("Environment check passed! Your architecture is structurally flawless.")
                
