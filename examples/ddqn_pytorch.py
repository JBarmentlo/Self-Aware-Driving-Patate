'''
file: ddqn.py
author: Felix Yu
date: 2018-09-12
original: https://github.com/flyyufelix/donkey_rl/blob/master/donkey_rl/src/ddqn.py
'''
import os
import sys
import random
import argparse
import signal
import uuid

import numpy as np
import gym
import cv2

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

from collections import deque
import gym_donkeycar
import collections

import torch
import torch.nn.functional as F

EPISODES = 10000
img_rows, img_cols = 80, 80
# Convert image into Black and white
img_channels = 4 # We stack 4 frames

class DQNAgent:

    def __init__(self, state_size, action_space, train=True):
        self.t = 0
        self.max_Q = 0
        self.train = train
        
        # Get size of state and action
        self.state_size = state_size
        self.action_space = action_space
        self.action_size = action_space

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        if (self.train):
            self.epsilon = 1.0
            self.initial_epsilon = 1.0
        else:
            self.epsilon = 1e-6
            self.initial_epsilon = 1e-6
        self.epsilon_min = 0.02
        self.batch_size = 64
        self.train_start = 100
        self.explore = 10000

        # Create replay memory using deque
        self.memory = deque(maxlen=10000)

        # Create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # Copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()


    def build_model(self):
        # padding P for same padding (rounded up)
        # P = ((S-1)*W-S+F)/2, with F = filter size, S = stride
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = img_channels, out_channels = 24, kernel_size = (5, 5), stride = (2, 2), padding = 42),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = 24, out_channels = 32, kernel_size = 5, stride = 2, padding = 42),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 2, padding = 42),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride = 2, padding = 42),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride = 2, padding = 42),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(80 * 80 * 64)
        )


    def rgb2gray(self, rgb):
        '''
        take a numpy rgb image return a new single channel image converted to greyscale
        '''
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


    def process_image(self, obs):
        obs = self.rgb2gray(obs)
        obs = cv2.resize(obs, (img_rows, img_cols))
        return obs


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        # self.target_model.set_weights(self.model.get_weights())