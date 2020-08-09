from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

from utils.env import StockMarketEnv
from utils.data_generator import DataGenerator
from utils import evaluate

eval_data_generator = DataGenerator().evaluation_generator()
models_data_generator = DataGenerator().models_generator()
is_having_new_models = True

while is_having_new_models:
    try:
        test_df, instrument_name = next(eval_data_generator)
        eval_env = DummyVecEnv([lambda: StockMarketEnv(test_df, is_training=False)])
        model_name = next(models_data_generator)
        model = PPO2.load(model_name)
        for i in range(1, 4):
            eval_name = instrument_name + '_0' + str(i)
            evaluate.evaluate_trained_model(eval_env, eval_name, model)

    except StopIteration:
        is_having_new_models = False
