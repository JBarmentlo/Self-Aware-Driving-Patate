import io

from Memory import DqnMemory
from S3 import S3
from Database import Database


class Agent():
    def __init__(self, config):
        self.config = config
        self.memory = self._init_memory(config.config_Memory)
    

    def _init_memory(self, config = None):
        pass


    def get_action(self, state, episode = 0):
        return (np.random.random((2)) - [0.5, 0]) * [6, 1]

    
    def train(self):
        pass


import math
import random
import numpy as np
import random
from collections import namedtuple, deque
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from utils import val_to_idx



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALogger = logging.getLogger("DQNAgent")
ALogger.setLevel(logging.INFO)
stream = logging.StreamHandler()
ALogger.addHandler(stream)

class  DQNAgent():
    def __init__(self, config):
        self.config = config
        self.memory = self._init_memory(config.config_Memory)
        self.model = DQN(config)
        self._init_S3(config.config_S3)
        if (self.config.load_model):
            self._load_model(self.config.model_to_load)
        self.model.to(device)
        self.target_model = DQN(config)
        self.target_model.to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.criterion = nn.MSELoss()
        self.update_target_model_counter = 0
        self.DB = Database(config.config_Database, self.S3)


    def save_modelo(self, file_name):
        ### TODO : in the future, maybe save more than just weights
        s3_name = f"{self.config.config_S3.model_folder}{file_name}"
        local_name = f"{self.config.local_model_folder}{file_name}"
        if self.config.S3_connection == True:
            buffer = io.BytesIO()
            torch.save(self.model.state_dict(), buffer)
            buffer.seek(0) # ! Reset read pointer. DOT NOT FORGET THIS, else all uploaded files will be empty!
            self.S3.upload_bytes(buffer, f"{s3_name}")
        else:
              torch.save(self.model.state_dict(), local_name)
              ALogger.info(f"Saving modelo locally in :{local_name}")


    def _load_model(self, path):
        try:
            input = path
            if self.config.S3_connection == True:
                bytes_obj = self.S3.get_bytes(path)
                input = io.BytesIO(bytes_obj)
            self.model.load_state_dict(torch.load(input, map_location=torch.device('cpu'))) ##! to be changed, just for deya
            self.model.eval()
            ALogger.info(f"Loading model from path: {path}")
        except Exception as e:
            ALogger.error(f"You tried loading a model from path: {path} and this error occured: {e}")


    def action_from_q_values(self, qs):
        ALogger.debug(f"Getting steering from qs: {qs}")
        bounds = self.config.action_space_boundaries[0]
        l = len(qs[0])
        idx = torch.argmax(qs[0])
        action = self.config.action_space[idx]
        return action


    def _update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def _init_memory(self, config = None):
        return DqnMemory(self.config.config_Memory)

    def _init_S3(self, config = None):
        self.S3 = None
        if self.config.S3_connection == True:
            self.S3 = S3(config)


    def _update_epsilon(self):
        if self.config.epsilon > self.config.epsilon_min:
            self.config.epsilon -= (self.config.initial_epsilon - self.config.epsilon_min) / self.config.steps_to_eps_min
            ALogger.info(f"Updating self epsilon to {self.config.epsilon}")



    def get_action(self, state, episode = 0):
        if np.random.rand() > self.config.epsilon :
            ALogger.debug(f"Not Random action being picked")
            action = self.action_from_q_values(self.model.forward(torch.Tensor(state[np.newaxis, :, :])))
            ALogger.debug(f"{action = }")
            return action
        else:
            ALogger.debug(f"Random action being picked")
            action = random.choice(self.config.action_space)
            ALogger.debug(f"{action = }")
            return action


    def train_model(self, x, y):
        #TODO: make this exist
        self.model.train()
        y_hat = self.model.forward(x)
        loss = self.criterion(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.eval()

    


    def replay_memory(self):
        #TODO : batches and batch_size as args
        if len(self.memory) < self.config.min_memory_size:
            return
        ALogger.info(f"Replay from memory {len(self.memory)}")
        
        dataloader = DataLoader(self.memory, batch_size=4,
                        shuffle=False, num_workers=1)

        for i, single_batch in enumerate(dataloader):
            if i > self.config.batches_number:
                return
            self.update_target_model_counter += 1
            self.optimizer.zero_grad()
            targets = []
            processed_states, actions, new_processed_states, rewards, dones = single_batch
            dones = ~dones

            qs_b = self.model.forward(processed_states)
            qss_b = self.target_model.forward(new_processed_states)
            qss_max_b, _ = torch.max(qss_b, dim = 1)

            for i, (action, reward, done) in enumerate(zip(actions, rewards, dones)):
                target = qs_b[i].clone()
                target = target.detach()
                a_idx = val_to_idx(action, self.config.action_space)
                target[a_idx] = reward + (done * self.config.discount * qss_max_b[i]) 
                targets.append(target)

            targets = torch.stack(targets)
            error = self.criterion(qs_b, targets)
            error.backward()
            self.optimizer.step()

            if (self.update_target_model_counter % self.config.target_model_update_frequency == 0):
                self._update_target_model()
            

    def train(self):
        pass



def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) / stride  + 1


Logger = logging.getLogger("DQN")
Logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
Logger.addHandler(stream)


class DQN(nn.Module):

    def __init__(self, config):
        super(DQN, self).__init__()
        in_channels = [*config.input_size][0]
        self.conv0 = nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding = 2) # (kernal_size - 1) / 2 for same paddind
        self.conv1 = nn.Conv2d(24, 32, kernel_size=5, stride=2, padding = 2) # (kernal_size - 1) / 2 for same paddind
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding = 1)
        # self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding = 2)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(1600, 512)
        output_size = 1
        for s in config.action_space_size:
            output_size *= s
        self.dense2 = nn.Linear(512, output_size)


        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        # def conv2d_size_out(size, kernel_size = 5, stride = 2):
        # 	return (size - (kernel_size - 1) - 1) // stride  + 1
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # linear_input_size = convw * convh * 32
        # self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        Logger.debug(f"Forward x: {x.shape}")
        x = x.to(device)
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        Logger.debug(f"conv1: {x.shape}")
        x = F.relu(self.conv2(x))
        Logger.debug(f"conv2: {x.shape}")
        x = F.relu(self.conv3(x))
        Logger.debug(f"conv3: {x.shape}")
        x = self.flatten(x)
        Logger.debug(f"Flat: {x.shape}")
        x = F.relu(self.dense1(x))
        Logger.debug(f"dense1: {x.shape}")
        x = self.dense2(x)
        Logger.debug(f"dense2: {x.shape}")
        return x