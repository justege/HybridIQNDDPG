
from stable_baselines3 import PPO
import gym
from stable_baselines3 import A2C

gym.register(
    id='ABC-v0',
    entry_point='gym.envs.classic_control:ABC'
)
env = gym.make("ABC-v0")

#env = VecNormalize(env, norm_obs=True, norm_reward=False)

# Define the TD3 agent
#model = PPO('MlpPolicy', env, verbose=1)

model = A2C("MlpPolicy", env=env,
    verbose=1,
    gamma=0.99,
    learning_rate=0.0007,
    normalize_advantage=True,
    n_steps=64,
    )

#model = PPO.load("test.zip", env=env, verbose=1)

# Train the agent on the Pendulum environment
model.learn(total_timesteps=int(1e7))

# Save the trained agent
model.save("A2C_2018")

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






