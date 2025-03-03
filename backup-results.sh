#!/bin/bash

# Specify the folder to monitor
MONITORED_FOLDER="results"
# Specify the backup directory
BACKUP_DIR="results_backup"

# Create the backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Define a file to store the checksum of the monitored folder
CHECKSUM_FILE="$BACKUP_DIR/$(basename "$MONITORED_FOLDER")_checksum.txt"

# Calculate the current checksum of the monitored folder
CURRENT_CHECKSUM=$(find "$MONITORED_FOLDER" -type f -exec sha256sum {} + | sort | sha256sum)

# Check if the checksum file exists
if [ -f "$CHECKSUM_FILE" ]; then
    # Read the previous checksum
    PREVIOUS_CHECKSUM=$(cat "$CHECKSUM_FILE")

    # Compare the current checksum with the previous checksum
    if [ "$CURRENT_CHECKSUM" != "$PREVIOUS_CHECKSUM" ]; then
        echo "Changes detected in $MONITORED_FOLDER. Creating backup..."

        # Create a timestamp
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

        # Create a zip archive of the monitored folder
        ZIP_NAME="$(basename "$MONITORED_FOLDER")_$TIMESTAMP.zip"
        zip -r "$BACKUP_DIR/$ZIP_NAME" "$MONITORED_FOLDER"

        # Update the checksum file
        echo "$CURRENT_CHECKSUM" > "$CHECKSUM_FILE"

        echo "Backup created: $BACKUP_DIR/$ZIP_NAME"
    else
        echo "No changes detected in $MONITORED_FOLDER."
    fi
else
    # If the checksum file does not exist, create the backup and the checksum file
    echo "No previous backup found. Creating initial backup..."

    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    ZIP_NAME="$(basename "$MONITORED_FOLDER")_$TIMESTAMP.zip"
    zip -r "$BACKUP_DIR/$ZIP_NAME" "$MONITORED_FOLDER"

    # Store the current checksum
    echo "$CURRENT_CHECKSUM" > "$CHECKSUM_FILE"

    echo "Initial backup created: $BACKUP_DIR/$ZIP_NAME"
fi