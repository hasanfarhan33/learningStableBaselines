from snake_converted import SnekEnv

env = SnekEnv()
episodes = 50 

for episode in range(episodes):
    done = False 
    obs = env.reset() 
    while not done: 
        random_action = env.action_space.sample()
        print("\naction: ", random_action)
        obs, reward, terminated, truncated, info = env.step(random_action)
        print("observation: ", obs)
        print("reward: ", reward)
        print("terminated: ", terminated)
        print("truncated: ", truncated)
        print("info: ", info)
        

