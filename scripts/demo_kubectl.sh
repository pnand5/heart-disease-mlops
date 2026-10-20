#!/bin/bash
# Screenshot helper: shows each command, then really runs it (Kubernetes deployment proof).
#   scripts/demo_kubectl.sh
cd "$(dirname "$0")/.." || exit 1
run() { printf '\n\033[1;32m$\033[0m %s\n' "$*"; eval "$*"; }

clear
run "kubectl --context docker-desktop -n heart-disease get deploy,pods,svc"
run "curl -s localhost:8080/health"; echo
run "cat scripts/sample_patient.json"
run "curl -s -X POST localhost:8080/predict -H 'Content-Type: application/json' -d @scripts/sample_patient.json"; echo
