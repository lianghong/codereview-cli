#!/bin/bash
# Sample shell script for testing

set -e

echo "Setting up environment..."

if [ -z "$HOME" ]; then
    echo "HOME is not set"
    exit 1
fi

echo "Done!"
