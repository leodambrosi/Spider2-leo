#!/bin/bash

# DeepSeek-DSPy Agent Runner
# Usage: ./run.sh [experiment_suffix]

set -e

# Get experiment suffix from command line or use default
EXPERIMENT_SUFFIX=${1:-"deepseek-dspy-test"}

# Check for DeepSeek API key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "Error: DEEPSEEK_API_KEY environment variable not set."
    echo "Please set your DeepSeek API key:"
    echo "  export DEEPSEEK_API_KEY=your_api_key_here"
    exit 1
fi

echo "Starting DeepSeek-DSPy Agent with experiment suffix: $EXPERIMENT_SUFFIX"
echo "Using model: deepseek-reasoner"

# Run the agent
python run.py \
    --model deepseek-reasoner \
    --suffix "$EXPERIMENT_SUFFIX" \
    --max_steps 20 \
    --max_memory_length 10 \
    --temperature 0.5 \
    --top_p 0.9 \
    --max_tokens 2000
    # Note: GEPA optimization is disabled by default (--no_gepa)
    # To enable experimental GEPA: add --use_gepa flag

echo "Agent execution completed."
echo "Results saved in: output/$EXPERIMENT_SUFFIX"