#!/bin/bash
# set -euo pipefail

# Define the container name
CONTAINER_NAME="ragent"
CONTAINER_VERSION="latest"

# Build the Docker image
echo "Building Docker image..."
export DOCKER_BUILDKIT=1
docker build --pull \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    -t $CONTAINER_NAME:$CONTAINER_VERSION \
    -f deployment/docker/Dockerfile \
    .

# Run the Docker container
echo "Running Docker container -> $CONTAINER_NAME:$CONTAINER_VERSION"
docker run -d \
    --name $CONTAINER_NAME \
    -p 4040:9090 \
    $CONTAINER_NAME:$CONTAINER_VERSION
