. docker/env.sh
docker build \
    -f docker/Dockerfile \
    -t $IMAGE_NAME \
    --force-rm=true \
    .