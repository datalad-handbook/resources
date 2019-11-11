#!/bin/sh

set -e

# Generate a singularity image with Neurodocker

docker run --rm kaczmarj/neurodocker:master generate singularity \
--base neurodebian:buster-non-free \
--pkg-manager apt \
--install datalad python3-tk python3-pandas python3-sklearn python3-seaborn \
> envs/Singularity.1
