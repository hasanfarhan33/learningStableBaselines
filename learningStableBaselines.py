import gymnasium as gym 
from stable_baselines3 import PPO, A2C
import os 
import time 

models_dir = f"models/A2C-{int(time.time())}"
logdir = f"logs/A2C-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make("LunarLander-v2", render_mode = "human")
observation, info = env.reset() 


# Loading a saved model --> loading a PPO model at 50k 
# models_dir = "models/A2C"
# model_path = f"{models_dir}/50000.zip"
# model = A2C.load(model_path, env=env)

# Loading a scratch model 
model = A2C("MlpPolicy", env, verbose = 1, tensorboard_log=logdir, device = "cuda")

# TRAINING MODEL
TIMESTEPS = 10000
for i in range(1, 100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

# episodes = 10 
# for ep in range(episodes):
#     observation, info = env.reset()
#     done = False 
    
#     while not done: 
#         env.render()
#         action, _ = model.predict(observation)
#         observation, reward, terminated, truncated, info = env.step(action)
    
 

env.close()