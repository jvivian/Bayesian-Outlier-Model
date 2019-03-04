#!/usr/bin/env bash

VERSION=1.0a6

# Delete build folder in case it exists
rm -r dist

# Build source distribution
python setup.py sdist

# Upload to PyPi
twine upload dist/*

# Clean up
rm -r dist
rm -r src/Bayesian_Outlier_Model.egg-info

# Build and push Docker
docker build -t jvivian/bayesian-outlier-model:${VERSION} ./docker
docker push jvivian/bayesian-outlier-model:${VERSION}
