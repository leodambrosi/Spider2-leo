# DeepSeek-Reasoner + DSPy GEPA Agent

An agent implementation using DeepSeek-Reasoner model with DSPy GEPA optimization for Spider 2.0 benchmarks.

## Features

- Uses DeepSeek-Reasoner as the primary LLM via DSPy LM interface
- Implements DSPy modules for structured reasoning with Chain-of-Thought
- Applies GEPA (Genetic Evolutionary Program Augmentation) optimization for prompt/program tuning
- Compatible with Spider 2.0 evaluation framework (Lite, Snow, DBT)
- Integrates with existing Spider-Agent environment and actions

## Installation

```bash
cd methods/deepseek-dspy-agent
pip install -r requirements.txt
```

## Configuration

### API Keys
Set your DeepSeek API key:
```bash
export DEEPSEEK_API_KEY=your_api_key_here
```

### Database Credentials
Ensure you have proper credentials for the databases you plan to use:
- BigQuery: `bigquery_credential.json` in spider-agent-lite directory
- Snowflake: `snowflake_credential.json` in spider-agent-lite directory
- SQLite: Local database files in `spider2-lite/resource/databases/spider2-localdb/`

### Spider 2.0 Setup
Follow the Spider 2.0-Lite setup instructions in the main README to download databases and configure credentials.

## Usage

### Basic Run
Run the agent on Spider 2.0-Lite with default settings:
```bash
python run.py --model deepseek-reasoner --suffix experiment_name
```

### With GEPA Optimization
Enable GEPA optimization (requires training examples):
```bash
python run.py --model deepseek-reasoner --use_gepa --gepa_iterations 10 --suffix gepa_experiment
```

### Task Filtering
Run only specific task types:
```bash
# BigQuery tasks only
python run.py --bq_only --suffix bq_test

# Local SQLite tasks only
python run.py --local_only --suffix local_test

# Snowflake tasks only
python run.py --sf_only --suffix sf_test
```

### Example Selection
Run specific examples:
```bash
# Run examples 0-9
python run.py --example_index "0-10" --suffix small_test

# Run specific examples by name
python run.py --example_name "bq001" --suffix single_test
```

## Architecture

### Core Components
1. `deepseek_agent.py` - Main agent class integrating DSPy with Spider environment
2. `gepa_optimizer.py` - GEPA optimization implementation using genetic algorithms
3. `run.py` - Entry point script with command-line interface

### DSPy Integration
- `DeepSeekReasonerLM`: DSPy LM wrapper for DeepSeek Reasoner API
- `ThoughtActionSignature`: DSPy signature for thought-action generation
- `DeepSeekDSPyAgent`: Main agent class using DSPy Chain-of-Thought

### GEPA Optimizer
The GEPA (Genetic Evolutionary Program Augmentation) optimizer uses genetic algorithms to evolve DSPy programs:
- **Population-based evolution** with selection, crossover, and mutation
- **Fitness evaluation** based on action parsing success
- **Configurable parameters**: population size, generations, mutation/crossover rates

## Example

See `example.py` for a mock demonstration:
```bash
python example.py
```

## Integration with Spider 2.0

The agent is designed to work with the existing Spider-Agent framework:

1. **Environment Compatibility**: Uses `Spider_Agent_Env` from spider-agent-lite
2. **Action Space**: Supports all standard actions (Bash, Terminate, SQL execution, file operations)
3. **Evaluation**: Compatible with Spider 2.0 evaluation suite

### Running Full Evaluation
```bash
# Run agent on Spider 2.0-Lite
python run.py --model deepseek-reasoner --suffix eval_run

# Convert results to submission format
python get_spider2lite_submission_data.py --experiment_suffix eval_run --results_folder_name ../../spider2-lite/evaluation_suite/eval_run

# Run evaluation
cd ../../spider2-lite/evaluation_suite
python evaluate.py --result_dir eval_run --mode exec_result
```

## Parameters

### Agent Parameters
- `--model`: Model name (default: deepseek-reasoner)
- `--max_steps`: Maximum steps per task (default: 20)
- `--max_memory_length`: History length (default: 10)
- `--temperature`, `--top_p`, `--max_tokens`: Generation parameters

### GEPA Parameters
- `--use_gepa`: Enable GEPA optimization (default: True)
- `--no_gepa`: Disable GEPA optimization
- `--gepa_iterations`: Number of generations (default: 10)

## Notes

- **DeepSeek API**: Requires valid API key and internet access
- **DSPy GEPA**: Optimization may require significant computational resources
- **Environment**: Docker may be required for full Spider-Agent functionality
- **Credential Security**: Never commit API keys or database credentials