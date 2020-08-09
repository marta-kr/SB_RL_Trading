import tensorflow as tf

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv


logdir = "./tensorboard/evaluation/"
file_writer = tf.summary.FileWriter(logdir)


def evaluate_trained_model(env, name, trained_model):
    assert isinstance(env, DummyVecEnv)
    assert isinstance(trained_model, PPO2)

    reward_summary = tf.Summary()
    tag = 'rewards/' + name
    reward_summary.value.add(tag=tag)

    profit_summary = tf.Summary()
    tag_profit = 'rewards/profit_' + name
    profit_summary.value.add(tag=tag_profit)

    obs = env.reset()
    for i in range(1500):
        action, _states = trained_model.predict(obs)
        obs, rewards, done, env_info = env.step(action)

        reward_summary.value[0].simple_value = rewards
        file_writer.add_summary(reward_summary, i)

        profit_summary.value[0].simple_value = env_info[0].get('profit')

        file_writer.add_summary(profit_summary, i)

        env.render()
