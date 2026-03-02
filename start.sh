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
    echo "Desired profiles are 'pc1' (preferably one with more VRAM than the pc2), and 'pc2' (preferably less VRAM than the pc1)."
    exit 1
fi

PROFILE=$1

if [ -z "$PROFILE" ]; then
    echo "Error: Missing profile argument."
    echo "Make sure to use './start.sh pc1' or './start.sh pc2'."
    exit 1
fi

docker compose --profile "$PROFILE" up -d
