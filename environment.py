import gymnasium as gym
import random
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from mpi4py import MPI

from task import RobotTask
from robot import Robot

from utils import sigmoid, scale_footstep_timing, convert_to_list

class RobotEnvironment(gym.Env):

    def __init__(self, action_repeat=400):
        self.seed()
        self.action_repeat = action_repeat
        self.robot = Robot(self.action_repeat)
        self.task = RobotTask()
        
        self.step_time = 0
        self.target_vel = None

        self.action_space = spaces.Box(
            np.array([0.0]*9),
            np.array([1.0]*9),
            dtype=np.float32)
        depth_space = spaces.Box(0.0, 1.0, shape=(1, 40, 40), dtype='float32')
        vector_space = spaces.Box(-np.inf, np.inf, shape=(16, ), dtype='float32')
        
        self.observation_space = spaces.Dict({"depth": depth_space, "vector": vector_space})
                                            
        self.reset()

        return

    def seed(self, seed=None):
        self.np_random, self.np_random_seed = seeding.np_random(seed)
        return [self.np_random_seed]

    def reset(self, seed = None, options = {}):
        self.step_time = 0
        self.target_vel = None
        obs, image = self.robot.reset_robot()

        return {"depth" : image.astype("float32"), "vector" : obs.astype("float32")}, {}
        # return obs, {}

    def step(self, action):
        act = sigmoid(action)
        
        self.robot.set_vel(self._get_target_vel())
        obs, img, safe = self.robot.step(act)

        rew = self.task.get_reward(obs)
        
        if safe:
            print("reward:", rew)

        self.step_time += 1

        done = False
        if self.step_time > 1000:
            done = True

        done = not safe or done

        return {"depth" : img.astype("float32") ,"vector" : obs.astype("float32")}, rew, done, False, {}

    def set_target_vel(self, vel):
        self.target_vel = vel

    def _get_target_vel(self):
        if self.target_vel is None:
            vels = [0.5, 0.6, 0.7, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]
            self.target_vel = [random.sample(vels, 1)[0] + random.random() * 0.1, 0, 0]
            
            print("vel = ", self.target_vel)
            
        return self.target_vel
