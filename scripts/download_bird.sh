#!/usr/bin/env bash
# Download and unpack the BIRD benchmark into data/bird/.
# Homepage: https://bird-bench.github.io/
set -euo pipefail

ROOT="${1:-data/bird}"
mkdir -p "$ROOT"
cd "$ROOT"

TRAIN_URL="https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
DEV_URL="https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"

if [ ! -d train ]; then
    echo "fetching BIRD train..."
    curl -L -o train.zip "$TRAIN_URL"
    unzip -q train.zip
    rm train.zip
    # BIRD train ships the databases as an inner zip; unpack it.
    if [ -f train/train_databases.zip ]; then
        (cd train && unzip -q train_databases.zip && rm train_databases.zip)
    fi
fi

if [ ! -d dev ]; then
    echo "fetching BIRD dev..."
    curl -L -o dev.zip "$DEV_URL"
    unzip -q dev.zip
    rm dev.zip
    if [ -f dev/dev_databases.zip ]; then
        (cd dev && unzip -q dev_databases.zip && rm dev_databases.zip)
    fi
fi

echo "BIRD ready under $(pwd)"
ls -la
