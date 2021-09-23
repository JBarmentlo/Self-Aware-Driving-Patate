import matplotlib.pyplot as plt
from S3 import S3
import io

from config import config

plt.figure()
plt.plot([1, 2])
plt.title("test")
# plt.savefig("hey")

my_s3 = S3("deyopotato")
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
my_s3.upload_bytes(buf, "model_cache/autoencoder/images_results/weshwesh")