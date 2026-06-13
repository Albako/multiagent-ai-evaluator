#!/bin/bash
# start.sh

ENV_FILE=".env"
EXAMPLE_FILE=".env.example"

if [ ! -f "$ENV_FILE" ]; then
    echo "Environment file not found."
    echo "Creating $ENV_FILE from $EXAMPLE_FILE..."
    cp "$EXAMPLE_FILE" "$ENV_FILE"
    echo "Initialization complete. Please configure your IPs in the $ENV_FILE file."
    echo "After configuring, run this script again with the desired profile."
    echo "Available profiles:"
    echo "  pc0          - Central API server (CPU only)"
    echo "  pc1          - Worker 1 Node (GPU)"
    echo "  pc2          - Worker 2 Node (GPU)"
    echo "  pc3          - Judge Node (GPU)"
    exit 1
fi

PROFILE=$1

if [ -z "$PROFILE" ]; then
    echo "Error: Missing profile argument."
    echo "Usage: ./start.sh <profile_name>"
    echo "Example: ./start.sh pc0"
    exit 1
fi

docker compose --profile "$PROFILE" up -d
