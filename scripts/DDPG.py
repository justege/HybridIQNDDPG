import gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

learning = True

gym.register(
    id='ABC-v0',
    entry_point='gym.envs.classic_control:ABC'
)

env = gym.make("ABC-v0")
env.seed(4)

gym.register(
    id='ABC-Test-v0',
    entry_point='gym.envs.classic_control:ABC_Test'
)

batches = 1
if learning == True:
    env = gym.make("ABC-v0")
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, learning_rate=0.0003)
    for batch in range(batches):
        model.learn(total_timesteps=int(400000))
        model.save("test_ddpg_" + str(batch))
else:
    env = gym.make("ABC-Test-v0")
    model = DDPG.load("test_ddpg.zip", env=env, verbose=1)
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






