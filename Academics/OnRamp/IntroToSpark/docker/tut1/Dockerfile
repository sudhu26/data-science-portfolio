FROM openjdk:8-alpine

# update ensures that Alpine has a fresh list of repos to dl from
# wget for downloading utilities
# tar for extracting files
# bash to run things
RUN apk --update add wget tar bash


# download SPark 2.4.3 based on Scala 2.11, Hadoop 2.7
RUN wget https://archive.apache.org/dist/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz


# extract spark from the zipped file, place in a folder, move to a folder called spark
# then remove the zipped file
RUN tar -xzf spark-2.4.0-bin-hadoop2.7.tgz && \
    mv spark-2.4.0-bin-hadoop2.7 /spark && \ 
    rm spark-2.4.0-bin-hadoop2.7.tgz