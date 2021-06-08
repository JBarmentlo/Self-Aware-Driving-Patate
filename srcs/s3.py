import boto3
import os

s3 = boto3.resource(
    service_name='s3',
    region_name= os.environ["REGION_NAME"],
    aws_access_key_id=os.environ["ACCESS_KEY"],
    aws_secret_access_key=os.environ["SECRET_KEY"]
)
bucket_name = 'deyopotato'
bucket = s3.Bucket(bucket_name)
