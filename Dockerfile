# FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
From nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04

WORKDIR /App

# BASIC INSTALL (APT + UPDATES)
RUN apt-get update \
	&& apt-get upgrade -y \
	&& apt-get install python3-dev -y \
	&& apt-get install python3.8-dev -y \
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

RUN git clone https://github.com/JBarmentlo/Self-Aware-Driving-Patate.git \
	&& cd Self-Aware-Driving-Patate \
	&& git checkout Reshape

# RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch==1.9.0 torchvision

RUN apt-get update
RUN apt-get install -y curl wget zsh
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN apt-get install direnv; echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc


WORKDIR /App/Self-Aware-Driving-Patate

# RUN apt-get install -y firefox
# RUN groupadd -g 1000 yup
# RUN useradd -d /home/yup -s /usr/bin/zsh -m yup -u 1000 -g 1000
# USER yup 

ENTRYPOINT /bin/zsh 
