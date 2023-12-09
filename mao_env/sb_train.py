import gymnasium as gym

from stable_baselines3 import DQN

from sb_env import SB3ActionMaskWrapper
from env import MaoEnv
from mao import Config
import supersuit as ss

env = SB3ActionMaskWrapper(MaoEnv(Config(4,["Alpha","Beta","Gamma","Delta"],52), render_mode="human"))
env = ss.vectorize_aec_env_v0(env, 1)
env = ss.concat_vec_envs_v1(env, 1, base_class='stable_baselines3')

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
# env = make_atari('BreakoutNoFrameskip-v4')

# model = DQN(CnnPolicy, env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("deepq_breakout")

# del model # remove to demonstrate saving and loading

# model = DQN.load("deepq_breakout")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()