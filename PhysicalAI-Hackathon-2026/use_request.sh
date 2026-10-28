#!/bin/sh
# Point the client at one of the saved requests before its slot:  ./use_request.sh qd44e20
# Each request's key file is kept as .request.<id>.json (private, like .request.json).
set -e
cd "$(dirname "$0")"
[ -f ".request.$1.json" ] || { echo "no saved request $1: have: $(ls .request.*.json | sed 's/\.request\.//; s/\.json//' | tr '\n' ' ')"; exit 1; }
cp ".request.$1.json" .request.json
echo "client now uses request $1"
