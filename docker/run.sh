. docker/env.sh
docker run \
  -dit \
  --gpus all \
  -v $PWD../:/workspace \
  -p 5000:5001 \
  --name $CONTAINER_NAME\
  --rm \
  --shm-size=2g \
  $IMAGE_NAME