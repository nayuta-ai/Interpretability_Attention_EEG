. docker/env.sh
docker run \
  -dit \
  -v $PWD:/workspace \
  -p 5000:5000 \
  --name $CONTAINER_NAME\
  --rm \
  --shm-size=2g \
  $IMAGE_NAME