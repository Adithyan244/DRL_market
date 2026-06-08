import math
import random
import matplotlib.pyplot as plt


class Hawkes_Process :

    def __init__(self,mu,alpha,beta):
        self.mu=mu
        self.alpha=alpha
        self.beta=beta
        self.curr_intensity=mu

    def step(self,event_occured):
        self.curr_intensity=self.mu+(self.curr_intensity-self.mu)*math.exp(-self.beta)
        if(event_occured):
            self.curr_intensity+=self.alpha
        return self.curr_intensity
    
    def generate_order(self):
        prob=min(0.99,self.curr_intensity)
        if random.random()<prob:
            return True
        return False


class limitOrderBook:

    def __init__(self):
        self.bids={}
        self.asks={}

    def add_limit_order(self,side,price,volume):
        #side is either buyer(bid) or seller(ask)
        if side=='bid':
            if price in self.bids:
                self.bids[price]+=volume
            else:
                self.bids[price]=volume
        elif side=='ask':
            if price in self.asks:
                self.asks[price]+=volume
            else:
                self.asks[price]=volume

    def get_best_bid(self):
        if not self.bids:
            return None
        return max(self.bids.keys())
    
    def get_best_ask(self):
        if not self.asks:
            return None
        return min(self.asks.keys())
    
    def get_mid_price(self):
        best_bid=self.get_best_bid()
        best_ask=self.get_best_ask()
        if best_bid is None or best_ask is None:
            return None
        return (best_bid+best_ask)/2.0
    
    def execute_market_order(self,side,volume):
        remaining_volume=volume
        if side == 'buy':
            while remaining_volume>0 and self.asks:
                best_ask=self.get_best_ask()
                available_volume=self.asks[best_ask]
                if remaining_volume>=available_volume:
                    remaining_volume-=available_volume
                    del self.asks[best_ask]
                else:
                    self.asks[best_ask]-=remaining_volume
                    remaining_volume=0
        elif side == 'sell':
            while remaining_volume>0 and self.bids:
                best_bid=self.get_best_bid()
                available_volume=self.bids[best_bid]
                if remaining_volume>=available_volume:
                    remaining_volume-=available_volume
                    del self.bids[best_bid]
                else:
                    self.bids[best_bid]-=remaining_volume
                    remaining_volume=0
        executed_volume=volume-remaining_volume
        return executed_volume


def simulate_random_market(book,num_steps):
    mid_price_history=[]
    intensity_history=[]
    hawkes=Hawkes_Process(mu=0.05,alpha=0.1,beta=0.15)
    for _ in range(num_steps):
        current_mid=book.get_mid_price()
        if current_mid is None:
            book.add_limit_order('bid',99.5,10)
            book.add_limit_order('ask',100.5,10)
            current_mid=100.0
        market_order=hawkes.generate_order()
        curr_lamda=hawkes.step(market_order)
        intensity_history.append(curr_lamda)
        if market_order:
            side=random.choice(['buy','sell'])
            volume=random.randint(10,50)
            book.execute_market_order(side,volume)
        else:
            if random.random()<0.6:
                side=random.choice(['bid','ask'])
                volume=random.randint(10,30)
                if side == 'bid':
                    price = round(current_mid-random.uniform(0.01,0.5),2)
                else:
                    price = round(current_mid+random.uniform(0.01,0.5),2)
                book.add_limit_order(side,price,volume)
        new_mid_price=book.get_mid_price()
        if new_mid_price is not None:
            mid_price_history.append(new_mid_price)
    return mid_price_history,intensity_history


if __name__ == "__main__":
    print("Initialising Limit Order Book....")
    book=limitOrderBook()
    print("Simulating 5000 orders....")
    price_history,intensity_history = simulate_random_market(book,num_steps=5000)
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,8),sharex=True)
    #plot 1: Mid Price
    ax1.plot(price_history,color='green',linewidth=1)
    ax1.set_title("Limit Order Mid Price")
    ax1.set_ylabel("Price ($)")
    ax1.grid(True,alpha=0.3)
    #plot 2: Hawkes process lambda
    ax2.plot(intensity_history, color='red', linewidth=1)
    ax2.set_title("Hawkes Process Intensity (Market Activity Level)")
    ax2.set_xlabel("Time Step (Ticks)")
    ax2.set_ylabel("λ(t) Probability")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

