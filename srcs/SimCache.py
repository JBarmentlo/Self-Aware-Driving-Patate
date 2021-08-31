import logging
import pickle
import os
import io

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
            self.folder = self.S3.config.simulator_folder
        else:
            self.folder = self.config.local_sim_folder	
        
        self.loading_counter	= 0
        self.list_files = [f"{self.config.sim_to_load}"]
        if self.config.sim_from_folder == True:
            if self.config.S3_connection == True:
                self.list_files = self.S3.get_folder_files(f"{self.folder}")
            else:
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
            buffer = io.BytesIO()
            pickle.dump(self.data, buffer)
            buffer.seek(0)
            ret = self.S3.upload_bytes(buffer, file_path)
        else:
            with open(file_path, "wb") as f:
                pickle.dump(self.data, f)
            Logger.info(f"Simulator Cache saved locally in file: {file_path}")
        self.upload_counter += 1
        self._reset()


    def load(self, path):
        if self.config.S3_connection == True:
            bytes_obj = self.S3.get_bytes(path)
            self.data = pickle.loads(bytes_obj)
        else:
            with open(path, "rb") as f:
                self.data = pickle.load(f)
            Logger.info(f"Loading data from file: {path}")
        self.loading_counter += 1


    def make_processed_memory(self):
        pass