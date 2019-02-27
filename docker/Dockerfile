FROM python:3

RUN pip install Bayesian-Outlier-Model==1.0a2

RUN mkdir /data
WORKDIR /data

ENTRYPOINT ["outlier-model"]
CMD ["--help"]
