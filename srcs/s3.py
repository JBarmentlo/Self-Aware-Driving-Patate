import boto3
import os
import pickle
import json


class S3:
    def __init__(self, region_name = None, aws_access_key_id = None, aws_secret_access_key = None):
        if region_name == None and aws_access_key_id == None and aws_secret_access_key == None:
            region_name= os.environ["REGION_NAME"]
            aws_access_key_id=os.environ["ACCESS_KEY"]
            aws_secret_access_key=os.environ["SECRET_KEY"]
        self.ressource = boto3.resource(
            service_name = 's3',
            region_name = region_name,
            aws_access_key_id = aws_access_key_id,
            aws_secret_access_key = aws_secret_access_key,
        )   

        self.bucket = self.ressource.Bucket(config.bucket_name)
    
    
    def pickle_upload(self, file_name, content):
        pickle_byte_obj = pickle.dumps(content)
        self.ressource.Object(config.bucket_name, file_name).put(Body=pickle_byte_obj)
    
    
    def read_s3_pkl_file(self, name):
        result = pickle.loads(self.bucket.Object(name).get()['Body'].read())
        return (result)
	

    def upload_json_file(self, file_name, content):
        json_byte_obj = json.dumps(content)
        self.ressource.Object(config.bucket_name, file_name).put(Body=json_byte_obj)