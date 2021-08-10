FROM ubuntu:18.04

WORKDIR /App

# BASIC INSTALL (APT + UPDATES)
RUN apt-get update \
	&& apt-get upgrade -y \
	&& apt install -y software-properties-common \
	&& add-apt-repository -y ppa:deadsnakes/ppa \
	&& apt install -y python3.8 git python3-pip zsh \
	# Reduce image size by deleting unnecessary cache
	&& rm -rf /var/lib/apt/lists/*

RUN python3.8 -m pip install --upgrade pip
RUN alias pip="echo 'carefull alias for pip 3.8';python3.8 -m pip"

# gym-donkeycar setup/install
COPY donkey_req.txt donkey_req.txt
RUN pip install -r donkey_req.txt
RUN git clone https://github.com/autorope/donkeycar \
	&& cd donkeycar \
	&& git checkout master \
	&& pip install -e .[pc]
RUN pip install git+https://github.com/tawnkramer/gym-donkeycar

# our project setup/install
COPY patata_req.txt patata_req.txt
RUN pip install -r patata_req.txt
RUN git clone https://github.com/JBarmentlo/Self-Aware-Driving-Patate.git \
	&& cd Self-Aware-Driving-Patate \
	&& git checkout Docker

WORKDIR /App/Self-Aware-Driving-Patate

ENTRYPOINT /bin/zsh