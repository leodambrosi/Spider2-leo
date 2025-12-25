# Spider 2.0 Project Context

## Project Overview
**Spider 2.0** is a comprehensive benchmark designed to evaluate Large Language Models (LLMs) on real-world enterprise text-to-SQL workflows. Unlike its predecessor (Spider 1.0), it focuses on complex, repository-level tasks involving:
*   **Large schemas** (>3000 columns).
*   **Multiple SQL dialects** (BigQuery, Snowflake, SQLite, DuckDB).
*   **Diverse operations** (transformations, analytics, data engineering).

The project is divided into three main evaluation settings:
1.  **Spider 2.0-Lite**: Uses BigQuery, Snowflake, and SQLite.
2.  **Spider 2.0-Snow**: Uses Snowflake exclusively (Tool-call optimized).
3.  **Spider 2.0-DBT**: Focuses on analytics engineering using dbt (DuckDB).

## Directory Structure

*   **`spider2-lite/`**: Contains data, instructions (`spider2-lite.jsonl`), and the evaluation suite for the Lite setting.
*   **`spider2-snow/`**: Contains data, instructions (`spider2-snow.jsonl`), and the evaluation suite for the Snowflake setting.
*   **`spider2-dbt/`**: Contains data, examples, and the evaluation suite for the DBT setting.
*   **`methods/`**: Contains reference agent implementations for running the benchmarks:
    *   `spider-agent-lite/`: Docker-based agent for Lite.
    *   `spider-agent-snow/`: Docker-based agent for Snow.
    *   `spider-agent-tc/`: **Docker-free**, tool-call based agent for Snow (Faster/Simpler).
    *   `spider-agent-dbt/`: Agent for DBT tasks.
*   **`assets/`**: Documentation and setup guides (especially for BigQuery/Snowflake credentials).

## Setup & Configuration

### 1. Prerequisites
*   **Python 3.11+** (Conda environment recommended).
*   **Docker**: Required for `spider-agent-lite`, `spider-agent-snow`, and `spider-agent-dbt`.
*   **Credentials**:
    *   **BigQuery**: JSON key file (for Lite).
    *   **Snowflake**: Username/Password (for Lite & Snow).

### 2. Credential Management
**CRITICAL**: Never commit credential files to version control.
*   Place `bigquery_credential.json` and `snowflake_credential.json` in the appropriate agent directories (e.g., `methods/spider-agent-lite/`).
*   Refer to `assets/Bigquery_Guideline.md` and `assets/Snowflake_Guideline.md` for obtaining keys.

## Building and Running

### Scenario A: Running Spider 2.0-Snow (Fastest, No Docker)
Use the `spider-agent-tc` method.

1.  **Setup**:
    ```bash
    cd methods/spider-agent-tc
    # Ensure credentials are in ./credentials folder
    ```
2.  **Run**:
    ```bash
    bash run.sh
    ```
3.  **Extract Submission**:
    ```bash
    python convert_to_submission_format.py <results_dir> <eval_suite_dir>
    ```

### Scenario B: Running Spider 2.0-Lite (Docker)
1.  **Install Dependencies**:
    ```bash
    cd methods/spider-agent-lite
    pip install -r requirements.txt
    ```
2.  **Setup Data**:
    ```bash
    # Downloads and unzips local databases
    python spider_agent_setup_lite.py
    ```
3.  **Run Agent**:
    ```bash
    export OPENAI_API_KEY=your_key
    python run.py --model gpt-4o -s test_run
    ```
4.  **Evaluate**:
    ```bash
    python get_spider2lite_submission_data.py --experiment_suffix test_run --results_folder_name ../../spider2-lite/evaluation_suite/results
    cd ../../spider2-lite/evaluation_suite
    python evaluate.py --result_dir results --mode exec_result
    ```

### Scenario C: Running Spider 2.0-DBT
1.  **Install Dependencies**:
    ```bash
    cd methods/spider-agent-dbt
    pip install -r requirements.txt
    ```
2.  **Setup Data**:
    *   Download `DBT_start_db.zip` and `dbt_gold.zip` (links in `spider2-dbt/README.md`).
    *   Run setup:
        ```bash
        cd ../../spider2-dbt
        python setup.py
        ```
3.  **Run Agent**:
    ```bash
    cd ../../methods/spider-agent-dbt
    python run.py --model gpt-4o --suffix test_run
    ```

## Development Conventions
*   **Submission Format**: Results are typically produced as CSVs or SQL files in specific directory structures to match the `evaluation_suite`.
*   **Gold Standards**: Some gold SQLs are provided in `evaluation_suite/gold` for debugging, but full evaluation relies on the provided scripts.
*   **File Paths**: Scripts often assume execution from their respective directories. Be mindful of relative paths (e.g., `../../spider2-lite`).
