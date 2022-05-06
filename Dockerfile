FROM python:3.9.2

ENV PYTHONUNBUFFERED 1

EXPOSE 8080
WORKDIR /app

COPY poetry.lock pyproject.toml ./

RUN apt-get update -y && \
    apt-get install build-essential libssl-dev cmake -y && \
    apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip && \
    pip install opencv-python scipy imutils attrs && \
    pip install --upgrade tensorflow && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

COPY . ./
ENV PYTHONPATH app
ENTRYPOINT ["python", "app/main.py"]