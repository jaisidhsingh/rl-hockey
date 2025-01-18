import time
import warnings
import numpy as np
import gymnasium as gym
from importlib import reload
import hockey.hockey_env as h_env
# disable warnings
warnings.simplefilter("ignore")

def first_look():
    np.set_printoptions(suppress=True)
    reload(h_env)

    env = h_env.HockeyEnv()
    obs, info = env.reset()
    obs_agent2 = env.obs_agent_two()

    for _ in range(600):
        env.render(mode="human")
        a1 = np.random.uniform(-1,1,4)
        a2 = np.random.uniform(-1,1,4)    
        obs, r, d, t, info = env.step(np.hstack([a1,a2]))    
        obs_agent2 = env.obs_agent_two()
        if d or t: 
            break
    
    env.close()

first_look()
