from tensorflow/tensorflow:2.2.2
# RUN pip install --upgrade pip
RUN apt-get update && apt install wget && apt install git
ENTRYPOINT bash