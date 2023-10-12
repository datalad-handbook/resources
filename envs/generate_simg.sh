#!/bin/sh

set -e

# Generate a singularity image with Neurodocker

docker run --rm kaczmarj/neurodocker:master generate singularity \
--base debian:stable-slim \
--pkg-manager apt \
--install python3-mne python3-numpy \
> envs/mne.1
