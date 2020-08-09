import enum
import random
import gym
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from gym import spaces

INIT_ACCOUNT_STATE = 5000
MAXIMUM_STEPS = 25000
COMMISSION = 0.0

MAX_ACCOUNT_BALANCE = 1000000
MAX_NUM_SHARES = 100
MAX_SHARE_PRICE = 5000
MAX_VOLUME = 1000000
MAX_PROFIT = 10000

EPISODE_LENGTH = 2000


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2


class StockMarketEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, stock_data, is_training=True):
        self.stock_data = stock_data

        self.action_space = gym.spaces.Discrete(n=len(Actions))

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1, 7), dtype=np.float16)

        self.balance = INIT_ACCOUNT_STATE
        self.possession_worth = INIT_ACCOUNT_STATE
        self.prev_possession_worth = INIT_ACCOUNT_STATE
        self.shares_held = 0
        self.profit = 0
        self.current_step = 0
        self.is_training = is_training
        self.chosen_action = Actions.Skip

    def _collect_observation(self):
        # norm_shares = 0.0
        # if self.shares_held > 0:
        #     norm_shares = self.shares_held / MAX_NUM_SHARES
        #
        # norm_profit = self.profit / MAX_PROFIT
        # norm_balance = self.balance / MAX_ACCOUNT_BALANCE
        #
        # observations = np.array([
        #     self.stock_data.loc[self.current_step, 'Open'] / MAX_SHARE_PRICE,
        #     self.stock_data.loc[self.current_step, 'High'] / MAX_SHARE_PRICE,
        #     self.stock_data.loc[self.current_step, 'Low'] / MAX_SHARE_PRICE,
        #     self.stock_data.loc[self.current_step, 'Close'] / MAX_SHARE_PRICE,
        #     self.stock_data.loc[self.current_step, 'Volume'] / MAX_VOLUME,
        #     norm_balance,
        #     norm_shares,
        #     norm_profit
        # ])
        #
        # return observations

        observations = np.array([
            self.stock_data.loc[self.current_step, 'Open'],
            self.stock_data.loc[self.current_step, 'High'],
            self.stock_data.loc[self.current_step, 'Low'],
            self.stock_data.loc[self.current_step, 'Close'],
            self.stock_data.loc[self.current_step, 'Volume'],
            self.balance,
            self.profit
        ])

        df = pd.DataFrame(observations)

        scaler = MinMaxScaler()
        scaler.fit(df)

        standardised = scaler.transform(df)
        observations_standardised = standardised.transpose()

        return observations_standardised

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.stock_data.loc[self.current_step, "High"], self.stock_data.loc[self.current_step, "Low"])

        assert isinstance(action, Actions)

        if action == Actions.Buy:
            # Buy
            shares_amount = int(self.balance / current_price)
            shares_cost = shares_amount * current_price

            self.shares_held += shares_amount
            self.balance -= shares_cost

        elif action == Actions.Close:
            # Sell
            money_for_shares = self.shares_held * current_price

            self.balance += money_for_shares
            self.shares_held = 0

        self.prev_possession_worth = self.possession_worth
        self.possession_worth = self.balance + self.shares_held * current_price
        self.profit = self.possession_worth - INIT_ACCOUNT_STATE

    def reset(self):
        # Reset the state of the environment to an initial state
        self.possession_worth = INIT_ACCOUNT_STATE
        self.prev_possession_worth = INIT_ACCOUNT_STATE
        self.balance = INIT_ACCOUNT_STATE
        self.shares_held = 0
        self.profit = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            0, len(self.stock_data.loc[:, 'Open'].values) - EPISODE_LENGTH - 1)

        return self._collect_observation()

    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Possession worth: {self.possession_worth}')
        print(f'Profit: {self.profit}')
        if self.chosen_action == Actions.Close:
            print(f'Chosen action: Close')
        elif self.chosen_action == Actions.Buy:
            print(f'Chosen action: Buy')
        elif self.chosen_action == Actions.Skip:
            print(f'Chosen action: Skip')
        print(f' ')

    def step(self, action_idx):
        action = Actions(action_idx)
        self.chosen_action = action
        self._take_action(action)

        self.current_step += 1

        reward = self.possession_worth - self.prev_possession_worth
        done = self._done()
        obs = self._collect_observation()
        profit_info = {'profit': self.profit}

        return obs, reward, done, profit_info

    def _done(self):
        if self.current_step > len(self.stock_data.loc[:, 'Open'].values)-2:
            return True
        return self.profit <= -100 and self.is_training

    def set_data(self, stock_data):
        self.stock_data = stock_data
