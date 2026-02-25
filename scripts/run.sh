#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
source ~/.bash_profile
conda activate llamaindex
python src/run.py "$@"