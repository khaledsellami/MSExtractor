FROM python:3.10-slim

RUN mkdir /service
WORKDIR /service
COPY . .
RUN python -m pip install --upgrade pip
RUN python -m pip install .
RUN rm -rf msextractor

EXPOSE 50060
ENTRYPOINT [ "python", "main.py" ]