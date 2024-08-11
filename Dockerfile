FROM python:3.12
LABEL authors="sallo"

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

ENV TZ=Europe/Moscow

RUN apt-get update && apt-get install -yy tzdata
RUN cp /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

RUN TZ=Europe/Moscow
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN chmod 777 .
#RUN chown -R root /etc/
COPY . .


