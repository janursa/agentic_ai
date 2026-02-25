#!/bin/bash
# Run the LlamaIndex agent
cd "$(dirname "$0")/.." || exit 1
source ~/.bash_profile
conda activate llamaindex
cd src
python run.py "$@"