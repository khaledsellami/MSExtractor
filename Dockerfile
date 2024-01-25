FROM python:3.6-slim

RUN mkdir /service
WORKDIR /service
COPY . .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

ENTRYPOINT [ "python", "main.py" ]