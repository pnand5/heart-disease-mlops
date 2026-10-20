#!/bin/bash
# Screenshot helper: build the image, run it in isolation and call it (Docker proof).
#   scripts/demo_docker.sh
cd "$(dirname "$0")/.." || exit 1
run() { printf '\n\033[1;32m$\033[0m %s\n' "$*"; eval "$*"; }

docker rm -f heart-demo >/dev/null 2>&1
clear
run "docker build --progress=plain -t heart-disease-api:latest . 2>&1 | tail -5"
run "docker images heart-disease-api"
run "docker run -d --name heart-demo -p 8001:8000 heart-disease-api:latest"
# wait (silently) until Docker's own health check reports healthy, so the status line below shows it
for _ in $(seq 1 60); do
  [ "$(docker inspect -f '{{.State.Health.Status}}' heart-demo 2>/dev/null)" = healthy ] && break
  sleep 1
done
run "docker ps --filter name=heart-demo --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'"
run "curl -s localhost:8001/health"; echo
run "curl -s -X POST localhost:8001/predict -H 'Content-Type: application/json' -d @scripts/sample_patient.json"; echo
run "curl -s -o /dev/null -w 'invalid input -> HTTP %{http_code}\n' -X POST localhost:8001/predict -H 'Content-Type: application/json' -d '{\"age\": -5}'"
run "docker rm -f heart-demo"
