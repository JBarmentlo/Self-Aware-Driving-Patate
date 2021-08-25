import boto3
import os
import pickle
import json
import logging

Logger = logging.getLogger("S3")
Logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
Logger.addHandler(stream)


class S3:
    def __init__(self, config, aws_access_key_id = None, aws_secret_access_key = None):
        self.config = config
        self.bucket_name = self.config.bucket_name
        if aws_access_key_id == None and aws_secret_access_key == None:
            aws_access_key_id=os.environ["ACCESS_KEY"]
            aws_secret_access_key=os.environ["SECRET_ACCESS_KEY"]
        try:
            self.client = boto3.client(
                service_name = 's3',
                aws_access_key_id = aws_access_key_id,
                aws_secret_access_key = aws_secret_access_key,
            )
            self.resource = boto3.resource(
                service_name = 's3',
                aws_access_key_id = aws_access_key_id,
                aws_secret_access_key = aws_secret_access_key,
            )
            Logger.info(f"Connection to S3 successful!")
        except Exception as e:
            Logger.error(f"Could not connect to S3:\n{e}")
    
    
    def download_file(self, s3_model_path, local_model_path):
        try:
            self.client.download_file(self.config.bucket_name, s3_model_path, local_model_path)
            Logger.info(f"File {s3_model_path} successfully downloaded to local path: {local_model_path}")
        except Exception as e:
            Logger.error(f"Could not download file {s3_model_path} from S3:\n{e}")



    def upload_file(self, local_model_path, s3_model_path):
        try:
            res = self.client.upload_file(local_model_path, self.config.bucket_name, s3_model_path)
            Logger.info(f"File {local_model_path} successfully uploaded to s3 path: {s3_model_path}")
        except Exception as e:
            Logger.error(f"Could not upload file {local_model_path} to S3:\n{e}")
    

    def upload_object(self, python_object, s3_model_path):
        try:
            pickle_byte_obj = pickle.dumps(python_object)
            self.resource.Object(self.config.bucket_name, s3_model_path).put(Body=pickle_byte_obj)
        except Exception as e:
            Logger.error(f"Could not upload python_object to S3:\n{e}")
        
