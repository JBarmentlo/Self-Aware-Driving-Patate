import io
import torch

import logging

MLogger = logging.getLogger(__file__)
MLogger.setLevel(logging.INFO)
stream = logging.StreamHandler()
MLogger.addHandler(stream)

class ModelCache():
    def __init__(self, S3 = None):
        self.S3 = S3


    def _save_local(self, model, path):
        torch.save(model.state_dict(), path)

        
    def _save_s3(self, model, path):
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        self.S3.upload_bytes(buffer, path)


    def save(self, model, path):
        if self.S3 != None:
            MLogger.info(f"Saving model to S3 path: {path}")
            self._save_s3(model, path)
        else:
            MLogger.info(f"Saving model to local path: {path}")
            self._save_local(model, path)

    
    def _load_torch(self, model, path):
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))) 
        model.eval()


    def load(self, model, path):
        if self.S3 != None:
            bytes_obj = self.S3.get_bytes(path)
            where = io.BytesIO(bytes_obj)
            MLogger.info(f"Loading model from S3 path: {path}")
        else:
            MLogger.info(f"Loading model from local path: {path}")
            where = path
        self._load_torch(model, where)