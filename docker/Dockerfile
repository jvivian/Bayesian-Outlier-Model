FROM continuumio/anaconda3

RUN apt-get update && apt-get install -y libhdf5-serial-dev build-essential gcc

RUN pip install Bayesian-Outlier-Model==1.0a10

RUN mkdir /data
WORKDIR /data

ENTRYPOINT ["outlier-model"]
CMD ["--help"]
