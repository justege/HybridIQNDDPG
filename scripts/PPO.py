import gym
from stable_baselines3 import PPO, A2C
import numpy as np

learning = False

gym.register(
    id='ABC-v0',
    entry_point='gym.envs.classic_control:ABC'
)

env = gym.make("ABC-v0")

gym.register(
    id='ABC-Test-v0',
    entry_point='gym.envs.classic_control:ABC_Test'
)

env = gym.make("ABC-Test-v0")

if learning == True:
    env = gym.make("ABC-v0")
    n_actions = env.action_space.shape[-1]
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    model.learn(total_timesteps=int(5e5))
    model.save("test_ddpg")
else:
    env = gym.make("ABC-Test-v0")
    model = A2C.load("/Users/egemenokur/PycharmProjects/Playground/A2C_2018.zip", env=env, verbose=1)
    # Test the agent

    obs = env.reset()
    total_reward = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(obs, rewards, done, info)
        total_reward =+ rewards
        if done:
            break

    print("Total reward:", total_reward)
# Close the environment






