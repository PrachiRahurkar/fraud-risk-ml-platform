#!/usr/bin/env bash
# Create Kafka topics inside the running Docker Kafka container.
# Usage: bash ml-infra-deployment-ops/kafka/topics.sh

set -euo pipefail

CONTAINER="${KAFKA_CONTAINER:-ml-infra-deployment-ops-kafka-1}"
BOOTSTRAP="localhost:9092"
PARTITIONS="${KAFKA_PARTITIONS:-3}"
REPLICATION="${KAFKA_REPLICATION:-1}"

TOPICS=(
  "fraud-risk.fund-events"
  "fraud-risk.predictions"
  "fraud-risk.labels"
)

for TOPIC in "${TOPICS[@]}"; do
  echo "Creating topic: $TOPIC"
  docker exec "$CONTAINER" kafka-topics \
    --bootstrap-server "$BOOTSTRAP" \
    --create \
    --if-not-exists \
    --topic "$TOPIC" \
    --partitions "$PARTITIONS" \
    --replication-factor "$REPLICATION"
done

echo ""
echo "Done. All topics:"
docker exec "$CONTAINER" kafka-topics --bootstrap-server "$BOOTSTRAP" --list | grep "fraud-risk"
