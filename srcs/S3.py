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
    def __init__(self, bucket_name, aws_access_key_id = None, aws_secret_access_key = None):
        self.bucket_name = bucket_name
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
    
    
    def download_file(self, s3_path, local_model_path):
        try:
            self.client.download_file(self.bucket_name, s3_path, local_model_path)
        except Exception as e:
            Logger.error(f"Could not download file {s3_path} from S3:\n{e}")



    def upload_file(self, local_model_path, s3_path):
        try:
            res = self.client.upload_file(local_model_path, self.bucket_name, s3_path)
        except Exception as e:
            Logger.error(f"Could not upload file {local_model_path} to S3:\n{e}")
    

    def upload_bytes(self, bytes_object, s3_path):
        try:
            self.client.upload_fileobj(bytes_object, self.bucket_name, s3_path)
            return True
        except Exception as e:
            Logger.error(f"Could not upload python_object to S3:\n{e}")
            return False
    
    
    def get_bytes(self, s3_path):
        try:
            s3_obj = self.resource.Object(self.bucket_name, s3_path)
            bytes_obj = s3_obj.get()['Body'].read()
            return (bytes_obj)
        except Exception as e:
            Logger.error(f"Could not get python_object from S3:\n{e}")
            return None
        
    
    def get_folder_files(self, prefix):
        list_files = []
        bucket = self.resource.Bucket(self.bucket_name)
        for object_summary in bucket.objects.filter(Prefix=prefix):
            split_path = object_summary.key.split("/")
            if len(split_path) > 1 and split_path[-1] != "":
                file_name = split_path[-1]
                list_files.append(prefix + file_name)
        return (list_files)

# #louis:
# 	def name_file(self, name):
# 		s3_name = f"{self.config.model_folder}{name}"
# 		return s3_name
		
