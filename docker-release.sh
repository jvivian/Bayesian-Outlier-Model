#!/usr/bin/env bash

VERSION=1.0a14

# Build and push Docker
docker build -t jvivian/bayesian-outlier-model:${VERSION} ./docker
docker push jvivian/bayesian-outlier-model:${VERSION}
