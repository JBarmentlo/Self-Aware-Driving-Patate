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
            Logger.info(f"Connection to S3 successful!")
        except Exception as e:
            Logger.error(f"Could not connect to S3:\n{e}")
    
    
    def download_file(self, s3_path, local_path):
        try:
            self.client.download_file(self.config.bucket_name, s3_path, local_path)
            Logger.info(f"File {s3_path} successfully downloaded to local path: {local_path}")
        except Exception as e:
            Logger.error(f"Could not download file {s3_path} from S3:\n{e}")



    def upload_file(self, local_path, s3_path):
        try:
            res = self.client.upload_file(local_path, self.config.bucket_name, s3_path)
            Logger.info(f"File {local_path} successfully uploaded to s3 path: {s3_path}")
        except Exception as e:
            Logger.error(f"Could not upload file {local_path} to S3:\n{e}")
