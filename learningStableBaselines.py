import gymnasium as gym 
from stable_baselines3 import PPO


env = gym.make("LunarLander-v2", render_mode = "human")
observation, info = env.reset() 

model = PPO("MlpPolicy", env, verbose = 1)
model.learn(total_timesteps=10000)

episodes = 10 
for ep in range(episodes):
    observation, info = env.reset()
    done = False 
    
    while not done: 
        env.render()
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    
env.close() 