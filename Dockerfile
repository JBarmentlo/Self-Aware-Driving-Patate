FROM ubuntu:18.04
WORKDIR /App
RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update
RUN apt install -y python3.7
RUN apt install -y git
RUN apt install -y python3-pip
RUN python3.7 -m pip install --upgrade pip
RUN alias pip='python3.7 -m pip'
COPY donkey_req.txt donkey_req.txt
RUN pip install -r donkey_req.txt
RUN git clone https://github.com/autorope/donkeycar && cd donkeycar && git checkout master && pip install -e .[pc]
RUN pip install git+https://github.com/tawnkramer/gym-donkeycar
# RUN apt install -y curl
COPY patata_req.txt patata_req.txt
RUN pip install -r patata_req.txt
RUN git clone https://github.com/JBarmentlo/Self-Aware-Driving-Patate.git && cd Self-Aware-Driving-Patate && git checkout Docker
# RUN apt-get update && apt install wget && apt install git
# RUN curl https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.1-Linux-aarch64.sh -o miniconda3.7.sh
# RUN bash miniconda3.7.sh
# RUN git clone https://github.com/autorope/donkeycar && cd donkeycar && git checkout master
# RUN conda env create -f install/envs/ubuntu.yml && conda activate donkey && pip install -e .[pc]
ENTRYPOINT bash