FROM ubuntu:18.04

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && pip3 install --upgrade pip

COPY requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt

COPY speech_emotion_recognition /speech_emotion_recognition

WORKDIR /

ENTRYPOINT ["python3"]
