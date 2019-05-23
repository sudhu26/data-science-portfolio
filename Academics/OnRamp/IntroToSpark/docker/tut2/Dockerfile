FROM openjdk:8-stretch

WORKDIR /main/

### debian packages
RUN apt-get update && apt-get install -y \
    bash \
    net-tools \
    software-properties-common \
    ssh \
    tar \
    tree \
    wget

### python
RUN apt-get install -y python3 python3-pip

# set default Python 3.5
RUN update-alternatives --install "/usr/bin/python" "python" "$(which python3)" 1

# additional packages
RUN pip3 install --upgrade pip setuptools
WORKDIR /requirements/
COPY requirements.txt /requirements
RUN pip3 install -r requirements.txt

### spark
# download SPark 2.4.3 based on Scala 2.11, Hadoop 2.7
RUN wget https://archive.apache.org/dist/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz

# extract spark from the zipped file, place in a folder, move to a folder called spark
# then remove the zipped file
RUN tar -xzf spark-2.4.0-bin-hadoop2.7.tgz && \
    mv spark-2.4.0-bin-hadoop2.7 /spark && \ 
    rm spark-2.4.0-bin-hadoop2.7.tgz


### ports
# expose for ssh
EXPOSE 22

# expose for spark
EXPOSE 7000-8000

# export for master web ui
EXPOSE 8080

# export for master web ui
EXPOSE 8081