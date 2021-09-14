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
        self.loading_counter	= 0
        self.S3					= S3
        self.list_files = [f"{self.config.load_name}"]
        if self.config.load_name.endswith("/*"):
            folder = self.config.load_name[0:-1]
            if S3 != None:
                self.list_files = self.S3.get_folder_files(folder)
            else:
                liste = os.listdir(folder)
                self.list_files = [folder + name for name in liste]
        self.nb_files_to_load = len(self.list_files)
        

    def _reset(self):
        self.datapoints_counter	= 0
        self.data = []


    def add_point(self, point):
        self.data.append(point)
        self.datapoints_counter += 1


    def _S3_upload(self, data, path):
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        buffer.seek(0)
        self.S3.upload_bytes(buffer, path)


    def _local_upload(self, data, path):
        with open(path, "wb") as f:    
            pickle.dump(data, f)

            
    def upload(self, path):
        if self.S3 != None:
            self._S3_upload(self.data, path)
            Logger.info(f"simcache data uploaded to S3 in : {path}")
        else:
            self._local_upload(self.data, path)
            Logger.info(f"simcache data uploaded locally in : {path}")
        self.upload_counter += 1
        self._reset()


    def _S3_load(self, path):
        bytes_obj = self.S3.get_bytes(path)
        data = pickle.loads(bytes_obj)
        return (data)


    def _local_load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return (data)
        


    def load(self, path):
        if self.S3 != None:
            self.data = self._S3_load(path)
            Logger.info(f"simcache data loaded from S3 path : {path}")
        else:
            self.data = self._local_load(path)
            Logger.info(f"simcache data loaded from local path : {path}")
        self.loading_counter += 1
        return (self.data)
