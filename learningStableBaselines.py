import gymnasium as gym 
from stable_baselines3 import A2C
import os 

models_dir = "models/A2C"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make("LunarLander-v2", render_mode = "human")
observation, info = env.reset() 

model = A2C("MlpPolicy", env, verbose = 1, tensorboard_log=logdir)

TIMESTEPS = 10000

for i in range(30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

'''
episodes = 10 
for ep in range(episodes):
    observation, info = env.reset()
    done = False 
    
    while not done: 
        env.render()
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    
 
'''
env.close()