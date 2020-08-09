import warnings
import tensorflow as tf
import logging

from datetime import datetime

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from utils.env import StockMarketEnv
from utils.data_generator import DataGenerator

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

tf.autograph.set_verbosity(0)
tf.get_logger().setLevel('INFO')
tf.get_logger().setLevel(logging.ERROR)

now = datetime.now()
logdir = "./tensorboard/" + now.strftime("%Y%m%d-%H%M%S") + "/"
file_writer = tf.summary.FileWriter(logdir)

data_generator = DataGenerator().training_generator()
train_df, instrument_name = next(data_generator)
training_env = DummyVecEnv([lambda: StockMarketEnv(train_df)])

model = PPO2(
    MlpLstmPolicy,
    env=training_env,
    tensorboard_log="./tensorboard/",
    nminibatches=1,
    gamma=0.9186054884987277,
    learning_rate=0.00010836268335278707,
    ent_coef=5.383544502833298e-06,
    cliprange=0.3473312894472549,
    noptepochs=3,
    lam=0.9593020393096344)

model.learn(
    total_timesteps=5,
    tb_log_name="training_logs")

is_having_new_instruments = True

while is_having_new_instruments:
    try:
        train_df, instrument_name = next(data_generator)
        training_env = DummyVecEnv([lambda: StockMarketEnv(train_df)])
        model.set_env(training_env)
        model.learn(
            total_timesteps=5,
            tb_log_name="training_logs")
    except StopIteration:
        is_having_new_instruments = False
