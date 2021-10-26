from srcs.Simulator import Simulator
from ModelCache import ModelCache
import torch
import logging
import time
import numpy as np
import json
import io

from RewardOpti import RewardOpti
from agents.Agent import DQNAgent
from Preprocessing import Preprocessing
from S3 import S3
import utils
from SimCache import SimCache
from Score import DistanceTracker

from agents.SAC import SoftActorCritic

from Memory import SACDataset
from agents.config import config as agent_config

Logger = logging.getLogger("NeuralPlayer")
Logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
Logger.addHandler(stream)


class NeuralPlayer():
    agent:          DQNAgent
    preprocessor:   Preprocessing
    simulator:      Simulator
    
    def __init__(self, config, env, simulator):
        self.config = config
        self.scores = [] # TODO: do we want to keep that?
        self.env = env
        self.simulator = simulator
        self._init_dataset(config.config_Datasets)
        self._init_agent(config.config_Agent)
        self._init_preprocessor(config.config_Preprocessing)
        self._init_reward_optimizer(self.config)
        self._save_config()


    def _init_dataset(self, config):
        self.S3 = None
        if self.config.config_Datasets.S3_connection == True:
            self.S3 = S3(self.config.config_Datasets.S3_bucket_name)
        if self.config.agent_name == "DQN":
            self.SimCache = SimCache(self.config.config_Datasets.ddqn.sim, self.S3)
        

    def _init_preprocessor(self, config_Preprocessing):
        self.preprocessor = Preprocessing(config = config_Preprocessing, S3=self.S3)


    def _init_agent(self, config_Agent):
        if self.config.agent_name == "SAC":
            self.agent = SoftActorCritic(agent_config)
            # self.agent = SoftActorCritic() ### TODO : rajouter config dedans
        elif self.config.agent_name == "DQN":
            self.agent = DQNAgent(config=config_Agent, S3=self.S3)

  
    def _init_reward_optimizer(self, config_NeuralPlayer):
        self.RO = RewardOpti(config_NeuralPlayer)
    
    
    def _train_agent(self):
        self.agent.train()
    
    
    def _save_config(self):
        if self.agent.config.data.saving_frequency > 0:
            config_dictionnary = {}
            for info in self.config:
                config_dictionnary[info] = self.config[info]
            file_path = f"{self.agent.config.data.save_name}{self.config.config_Datasets.config_extension}"
            if self.S3 != None:
                json_obj = json.dumps(config_dictionnary).encode('UTF-8')
                bytes_obj = io.BytesIO(json_obj)
                bytes_obj.seek(0)
                self.S3.upload_bytes(bytes_obj, file_path)
            else:
                with open(file_path, "w") as f:
                    json.dump(config_dictionnary, f)
                    Logger.info(f"Config information saved in file: {file_path}")


    def add_simcache_point(self, datapoint, e):
        if self.SimCache.datapoints_counter + 1 > self.agent.config.sim.size:
            self.SimCache.upload(f"{self.agent.config.sim.save_name}{e}")
        self.SimCache.add_point(datapoint)
                

    def train_agent_from_SimCache(self):
        Logger.info(f"Training agent from SimCache database")
        while self.SimCache.loading_counter < self.SimCache.nb_files_to_load:
            path = self.SimCache.list_files[self.SimCache.loading_counter]
            self.SimCache.load(path)
            infos = self.SimCache.data[0][5]
            self.Score = DistanceTracker(infos["pos"], infos["cte"])
                        
            for datapoint in self.SimCache.data:
                state, action, new_state, reward, done, infos = datapoint
                done = self._is_over_race(infos, done)
                reward = self.RO.sticks_and_carrots(action, infos, done)
                [action, reward] = utils.to_numpy_32([action, reward])
                self.Score.next(infos["pos"], infos["cte"])
                processed_state, new_processed_state = self.preprocessor.process(state), self.preprocessor.process(new_state)
                self.agent.memory.add(processed_state, action, new_processed_state, reward, done)

            for _ in range(self.config.replay_memory_batches):
                self.agent.replay_memory()
            
            if (self.agent.config.data.saving_frequency != 0):
                self.agent.ModelCache.save(self.agent.model, f"{self.agent.config.data.save_name}{self.SimCache.loading_counter}")
        


    def _is_over_race(self, infos, done):
        cte = infos["cte"]
        cte_corr = cte + self.config.cte_offset
        if (done):
            return True

        if (abs(cte) > 100):
            return True
        
        if (abs(cte_corr) > self.config.cte_limit):
            return True

        return False


    def get_action(self, state):
        return self.agent.get_action(self.preprocessor.process(state))


    def add_score(self, iteration):
        self.scores.append(iteration)

    def get_score(self):
        self.scores[len(self.scores) - 1]


    def do_races_ddqn(self, episodes):
        Logger.info(f"Doing {episodes} races.")
        for e in range(1, episodes + 1):
            Logger.info(f"\nepisode {e}/{episodes}")
            self.RO.new_race_init(e)
            
            self.simulator = utils.fix_cte(self.simulator)
            self.env = self.simulator.env

            state, reward, done, infos = self.env.step([0, 0])

            self.Score = DistanceTracker(infos["pos"], infos["cte"]) ## TODO: check that

            processed_state = self.preprocessor.process(state)
            done = self._is_over_race(infos, done)
            Logger.debug(f"Initial CTE: {infos['cte']}")
            iteration = 0
            while (not done):

                action = self.agent.get_action(processed_state, e)
                Logger.debug(f"action: {action}")
                new_state, reward, done, infos = self.env.step(action)
                if self.agent.config.sim.save == True:
                    self.add_simcache_point([state, action, new_state, reward, done, infos], e)
                self.Score.next(infos["pos"], infos["cte"]) ### TODO check that
                new_processed_state = self.preprocessor.process(new_state)
                done = self._is_over_race(infos, done)
                reward = self.RO.sticks_and_carrots(action, infos, done)
                [action, reward] = utils.to_numpy_32([action, reward])
                self.agent.memory.add(processed_state, action, new_processed_state, reward, done)
                processed_state = new_processed_state
                Logger.debug(f"cte:{infos['cte'] + 2.25}")
                iteration += 1
            
            self.add_score(iteration)
            self.agent._update_epsilon()
            if (e % self.config.replay_memory_freq == 0):
                for _ in range(self.config.replay_memory_batches):
                    self.agent.replay_memory()


            if (self.agent.config.data.saving_frequency != 0 and
                (e % self.agent.config.data.saving_frequency == 0 or e == self.config.episodes)):
                self.agent.ModelCache.save(self.agent.model, f"{self.agent.config.data.save_name}{e}")
        
        
        if self.agent.config.sim.save == True:
            self.SimCache.upload(f"{self.agent.config.sim.save_name}{e}")
        if self.config.data.save_score == True:
            self.Score.upload(f"{self.agent.config.data.save_name}{e}")
        Logger.info(f"Total distance: {self.Score.total_distance}")
        self.env.reset()
        return
    
    

    def do_races_sac(self, episodes):
        memory = SACDataset()
        print(f"Doing {episodes} races.")
        scores = []
        for e in range(1, episodes + 1):
            print(f"\nepisode {e}/{episodes}")
            # print(f"memory size = {len(self.agent.memory)}")
            self.RO.new_race_init(e)

            self.simulator = utils.fix_cte(self.simulator)
            self.env = self.simulator.env

            state, reward, done, infos = self.env.step([0, 0])
            processed_state = self.preprocessor.process(state)
            done = self._is_over_race(infos, done)
            print(f"Initial CTE: {infos['cte']}")
            iteration = 0
            while (not done):

                action = self.agent.get_action(processed_state)[0].numpy()
                # print(f"True {action = }")
                # print(f"action: st {int(action[0] * 100)/100:7.2} th {(action[1] * 100)/100:7.2}")
                new_state, reward, done, infos = self.env.step(action)
                # self.agent.add_simcache_point([state, action, new_state, reward, done, infos])
                new_processed_state = self.preprocessor.process(new_state)
                # print(f"{new_processed_state.shape = }")
                done = self._is_over_race(infos, done)

                reward = self.RO.sticks_and_carrots(action, infos, done)

                # [action, reward] = utils.to_numpy_32([action, reward])

                # print(f"{new_processed_state[0].shape = }")
                # print(f"{action = }")
                current_action = torch.tensor(action)
                # print(f"{current_action = }")
                memory.add(processed_state[0], current_action,
                           new_processed_state[0], reward, int(done))

                processed_state = new_processed_state
                # print(f"cte:{infos['cte'] + 2.25}")
                iteration += 1

            self.add_score(iteration)
            print(f"Episode len: {self.scores[-1]}")

            if (e % self.config.replay_memory_freq == 0):

                print("Training")
                # scores.append(len(memory))
                if self.agent.train(memory):
                    memory = SACDataset()
                print(self.scores)

        self.env.reset()
        return

    def do_races(self, episodes):
        if self.config.agent_name == "SAC":
            self.do_races_sac(episodes)
        elif self.config.agent_name == "DQN":
            self.do_races_ddqn(episodes)
