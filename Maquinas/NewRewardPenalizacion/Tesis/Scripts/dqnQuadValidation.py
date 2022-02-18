# --------------------------------------------------------------
# ------------------------- Libraries --------------------------
# --------------------------------------------------------------
import time
import os
import warnings
import math as m
import random
import cv2
import numpy as np
import gym
import matplotlib.pyplot as plt
import sim
import imutils 
from gym import spaces
import datetime
from QuadEnviroment2  import quadEnviSim
import csv



with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.dqn.policies import  MlpPolicy
    from stable_baselines3 import DQN
    #from stable_baselines.common.callbacks import BaseCallback
    #from stable_baselines.bench import Monitor
    #from stable_baselines.results_plotter import load_results, ts2xy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.callbacks import BaseCallback

# --------------------------------------------------------------
# ------------------------- Constants --------------------------
# --------------------------------------------------------------
# Physical Boundaries
MAX_X = 2.5
MIN_X = -2.5
MAX_Y = 5
MIN_Y = -2.5


def main_test():
    # Create log dir
    log_dir = "C:/Users/KSCC/Documents/Tesis/results/logs/EntrenamientoCompletado/"
    # Pepper Training Env
    env = quadEnviSim()
    # Load model
    model = DQN.load(log_dir + "20220106-235245.zip")
    # Enjoy trained agent
    obs = env.reset()
    i = 0
    episodes = 0
    num_time_steps = 0
    with open('20220106-235245.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "endCause", "numTimeSteps"])
        while episodes<300:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            num_time_steps += 1
            if done:
                obs = env.reset()
                writer.writerow([episodes + 1,info[1], num_time_steps + 1])
                num_time_steps = 0
                episodes +=1 

            i+= 1
if __name__ == "__main__":
    # Main function
    main_test()
