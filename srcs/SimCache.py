import logging
import pickle
import os

import utils

Logger = logging.getLogger("SimCache")
Logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
Logger.addHandler(stream)

class SimCache():
    def __init__(self, config, S3 = None):
        self.config				= config
        self.data				= []
        self.datapoints_counter	= 0
        self.upload_counter		= 0
        self.S3					= S3
        if self.config.S3_connection == True:
            self.folder = self.S3.similator_folder
        else:
            self.folder = self.config.local_sim_folder	
        
        self.loading_counter	= 0
        self.list_files = [f"{self.folder}{self.config.sim_to_load}"]
        if self.config.sim_from_folder == True:
            self.list_files = os.listdir(f"{self.folder}")
        self.nb_files_to_load = len(self.list_files)


    def _reset(self):
        self.datapoints_counter	= 0
        self.data = []


    def add_point(self, point):
        self.data.append(point)
        self.datapoints_counter += 1


    def upload(self, file_path = None):
        if file_path == None:
            file_path = f"{self.folder}{self.config.sim_infos_name}{self.upload_counter}"
        if self.config.S3_connection == True:
            self.S3.upload_bytes(self.data, file_path)
        else:
            with open(file_path, "wb") as f:
                pickle.dump(self.data, f)
            Logger.info(f"Simulator Cache saved locally in file: {file_path}")
            self.upload_counter += 1
        self._reset()


    def load(self, path = None):
        if path == None:
            path = self.folder + self.list_files[self.loading_counter]
        if self.config.S3_connection == True:
            self.data = self.S3.get_bytes(path)
        else:
            with open(path, "rb") as f:
                self.data = pickle.load(f)
            Logger.info(f"Loading data from file: {path}")
            self.loading_counter += 1
        if self.loading_counter == self.nb_files_to_load:
            return(True)
        return (False)


    def make_processed_memory(self):
        pass