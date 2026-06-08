import math
import random
from lob_simulator import Hawkes_Process,limitOrderBook

def run_as_baseline():
    print("Initialising Avallanda Stoikov Market Maker")
    book=limitOrderBook()
    hawkes=Hawkes_Process(mu=0.05,alpha=0.10,beta=0.15)
    book.add_limit_order('bid',99.50,50)
    book.add_limit_order('ask',100.50,50)
    gamma=0.05
    sigma=0.1
    T=1.0
    kappa=25
    max_steps=2000
    inventory=0
    balance=0.0
    for step in range(max_steps):
        time_fraction=step/max_steps
        time_left=T-time_fraction
        current_mid=book.get_mid_price() or 100.0
        reservation_price=current_mid-inventory*sigma*sigma*gamma*time_fraction
        optimal_spread=gamma*sigma*sigma*time_left+(2/gamma)*math.log(1+gamma/kappa)
        my_bid_price=round(reservation_price-(optimal_spread)/2,2)
        my_ask_price=round(reservation_price+(optimal_spread)/2,2)
        market_order_arrived=hawkes.generate_order()
        hawkes.step(event_occured=market_order_arrived)
        if market_order_arrived:
            side = random.choice(['buy', 'sell'])
            volume = random.randint(10, 50)
            best_bid = book.get_best_bid() or 99.50
            best_ask = book.get_best_ask() or 100.50
            if side == 'sell' and my_bid_price >= best_bid:
                inventory += 1
                balance -= my_bid_price
            elif side == 'buy' and my_ask_price <= best_ask:
                inventory -= 1
                balance += my_ask_price
            book.execute_market_order(side, volume)
        else:
            if random.random() < 0.60:
                maker_side = random.choice(['bid', 'ask'])
                maker_vol = random.randint(10, 30)
                if maker_side == 'bid':
                    book.add_limit_order('bid',round(current_mid-random.uniform(0.01,0.5),2),maker_vol)
                else:
                    book.add_limit_order('ask',round(current_mid+random.uniform(0.01,0.5),2),maker_vol)
        if(inventory>100 or inventory<-100):
            print(f"AS Model went bankrupt at step {step} with inventory {inventory}")
            break
        final_mid=book.get_mid_price() or 100.0
        final_pnl=balance+inventory*final_mid
        inventory_risk=0.01*(inventory**2)
        final_reward=final_pnl-inventory_risk
        print("AS Results incoming....")
        print(f"Final Inventory: {inventory} shares")
        print(f"Final raw pnl_value: {final_pnl}")
        print(f"Final reward {final_reward}")


if __name__ == "__main__":
    run_as_baseline()