#!/bin/bash

if [ -f "$1" ]; then
    echo "File downloaded successfully."
else
    echo "File download failed."
    exit 1
fi
