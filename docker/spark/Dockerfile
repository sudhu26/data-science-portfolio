FROM openjdk:8-stretch

### debian packages
RUN apt-get update && apt-get install -y \
    bash \
    nano \
    net-tools \
    software-properties-common \
    ssh \
    tar \
    tree \
    wget

### spark

## move files from downloads folder at the host to /home in the container
WORKDIR /home/
# COPY /Users/petersontylerd/Downloads/spark-2.4.3-bin-hadoop2.7.tgz /home/
# COPY /Users/petersontylerd/Downloads/* /home/
# COPY downloads/* /home/

# download Spark 2.4.3 based on Scala 2.11, Hadoop 2.7
RUN wget https://www-eu.apache.org/dist/spark/spark-2.4.3/spark-2.4.3-bin-hadoop2.7.tgz


RUN mkdir -p /usr/local/spark-2.4.3
RUN tar -zxf spark-2.4.3-bin-hadoop2.7.tgz -C /usr/local/spark-2.4.3/
RUN rm spark-2.4.3-bin-hadoop2.7.tgz
RUN update-alternatives --install "/usr/sbin/start-master" "start-master" "/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7/sbin/start-master.sh" 1
RUN update-alternatives --install "/usr/sbin/start-slave" "start-slave" "/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7/sbin/start-slave.sh" 1
RUN update-alternatives --install "/usr/sbin/start-slaves" "start-slaves" "/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7/sbin/start-slaves.sh" 1
RUN update-alternatives --install "/usr/sbin/start-all" "start-all" "/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7/sbin/start-all.sh" 1
RUN update-alternatives --install "/usr/sbin/stop-all" "stop-all" "/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7/sbin/stop-all.sh" 1
RUN update-alternatives --install "/usr/sbin/stop-master" "stop-master" "/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7/sbin/stop-master.sh" 1
RUN update-alternatives --install "/usr/sbin/stop-slaves" "stop-slaves" "/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7/sbin/stop-slaves.sh" 1
RUN update-alternatives --install "/usr/sbin/stop-slave" "stop-slave" "/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7/sbin/stop-slave.sh" 1
RUN update-alternatives --install "/usr/sbin/spark-daemon.sh" "spark-daemon.sh" "/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7/sbin/spark-daemon.sh" 1
RUN update-alternatives --install "/usr/sbin/spark-config.sh" "spark-config.sh" "/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7/sbin/spark-config.sh" 1
RUN update-alternatives --install "/usr/bin/spark-shell" "spark-shell" "/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7/bin/spark-shell" 1
RUN update-alternatives --install "/usr/bin/spark-class" "spark-class" "/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7/bin/spark-class" 1
RUN update-alternatives --install "/usr/bin/spark-sql" "spark-sql" "/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7/bin/spark-sql" 1
RUN update-alternatives --install "/usr/bin/spark-submit" "spark-submit" "/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7/bin/spark-submit" 1
RUN update-alternatives --install "/usr/bin/pyspark" "pyspark" "/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7/bin/pyspark" 1
RUN update-alternatives --install "/usr/bin/load-spark-env.sh" "load-spark-env.sh" "/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7/bin/load-spark-env.sh" 1
ENV SPARK_HOME="/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7"

# RUN cp conf/log4j.properties.template conf/log4j.properties

### python
RUN apt-get install -y python3 python3-pip

# set default Python 3.5
RUN update-alternatives --install "/usr/bin/python" "python" "$(which python3)" 1

# additional packages
RUN pip3 install --upgrade pip setuptools
WORKDIR /requirements/
COPY requirements.txt /requirements
RUN pip3 install -r requirements.txt

# set up ipython shell to fire with pyspark
ENV PYSPARK_DRIVER_PYTHON=ipython

# reset working directory
WORKDIR /home/

### ports
# expose for ssh
EXPOSE 22

# expose for spark
EXPOSE 7000-8000

# export for master web ui
EXPOSE 8080

# export for master web ui
EXPOSE 8081