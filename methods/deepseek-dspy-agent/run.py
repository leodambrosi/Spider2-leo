import argparse
import datetime
import json
import logging
import os
import random
import sys
import glob

# Add spider-agent-lite to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
spider_agent_lite_path = os.path.join(current_dir, "../spider-agent-lite")
if spider_agent_lite_path not in sys.path:
    sys.path.insert(0, spider_agent_lite_path)

from tqdm import tqdm

from spider_agent.envs.spider_agent import Spider_Agent_Env
from deepseek_agent import DeepSeekDSPyAgent

# Load environment variables from .env file
try:
    import dotenv
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, "../../.env")
    if os.path.exists(env_path):
        dotenv.load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on environment variables

# Logger Configs
logger = logging.getLogger("deepseek_dspy_agent")
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler(os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8")
debug_handler = logging.FileHandler(os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8")
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(os.path.join("logs", "sdebug-{:}.log".format(datetime_str)), encoding="utf-8")

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s")
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("deepseek_dspy_agent"))
sdebug_handler.addFilter(logging.Filter("deepseek_dspy_agent"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepSeek Reasoner + DSPy GEPA agent on Spider 2.0 benchmarks"
    )

    # Agent parameters
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--max_memory_length", type=int, default=10)
    parser.add_argument("--suffix", '-s', type=str, default="deepseek-dspy")
    parser.add_argument("--model", type=str, default="deepseek-reasoner")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--stop_token", type=str, default=None)

    # DSPy/GEPA parameters
    parser.add_argument("--use_gepa", action="store_true", default=False, help="Use GEPA optimization (experimental, minimal optimization)")
    parser.add_argument("--no_gepa", action="store_false", dest="use_gepa", help="Disable GEPA optimization (default)")
    parser.add_argument("--gepa_iterations", type=int, default=10, help="GEPA optimization iterations (if use_gepa)")

    # Task configuration
    parser.add_argument("--test_path", "-t", type=str, default="../../spider2-lite/spider2-lite.jsonl")
    parser.add_argument("--example_index", "-i", type=str, default="all", help="index range of examples, e.g., '0-10', '2,3', 'all'")
    parser.add_argument("--example_name", "-n", type=str, default="", help="name of specific example to run")
    parser.add_argument("--overwriting", action="store_true", default=False)
    parser.add_argument("--retry_failed", action="store_true", default=False)

    # Output configuration
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--bq_only", action="store_true")
    parser.add_argument("--local_only", action="store_true")
    parser.add_argument("--dbt_only", action="store_true")
    parser.add_argument("--sf_only", action="store_true")

    args = parser.parse_args()
    return args


def test(args: argparse.Namespace) -> None:
    """Run agent on test examples."""
    scores = []

    # Log arguments
    logger.info("Args: %s", args)

    # Create experiment ID
    if args.suffix == "":
        logger.warning("No suffix provided, using model name as experiment ID.")
        experiment_id = args.model.split("/")[-1]
    else:
        experiment_id = args.model.split("/")[-1] + "-" + args.suffix

    if args.plan:
        experiment_id = f"{experiment_id}-plan"

    # Load task configurations
    if not (os.path.exists(args.test_path) and (args.test_path.endswith(".jsonl") or args.test_path.endswith(".json"))):
        raise ValueError(f"Invalid test_path: {args.test_path}. File must exist and have .json or .jsonl extension.")

    with open(args.test_path, "r") as f:
        if args.test_path.endswith(".jsonl"):
            task_configs = [json.loads(line) for line in f]
        else:
            task_configs = json.load(f)

    # Filter tasks if needed
    if args.example_name != "":
        task_configs = [task for task in task_configs if args.example_name in task.get("instance_id", "")]
    elif args.example_index != "all":
        if "-" in args.example_index:
            start, end = map(int, args.example_index.split("-"))
            task_configs = task_configs[start:end]
        else:
            indices = list(map(int, args.example_index.split(",")))
            task_configs = [task_configs[i] for i in indices]

    # Determine valid task types
    valid_types = set()
    if args.local_only:
        valid_types.add('local')
    if args.bq_only:
        valid_types.add('bq')
    if args.sf_only:
        valid_types.add('sf')
    if args.dbt_only:
        valid_types.add('dbt')

    # Create logs and output directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each task
    for task_config in tqdm(task_configs, desc="Processing tasks"):
        instance_id = task_config.get("instance_id", "unknown")
        full_instance_id = experiment_id + "/" + instance_id
        output_dir = os.path.join(args.output_dir, full_instance_id)
        result_json_path = os.path.join(output_dir, "spider/result.json")

        # Determine task type
        task_type = None
        if instance_id.startswith("bq") or instance_id.startswith("ga"):
            task_type = 'bq'
            task_config['type'] = 'Bigquery'
        elif instance_id.startswith("local"):
            task_type = 'local'
            task_config['type'] = 'Local'
        elif instance_id.startswith("sf"):
            task_type = 'sf'
            task_config['type'] = 'Snowflake'
        else:
            task_type = 'dbt'

        # Skip if filtering by type
        if valid_types and task_type not in valid_types:
            continue

        # Skip if already exists and not overwriting
        if not args.overwriting and os.path.exists(result_json_path):
            logger.info("Skipping %s", full_instance_id)
            continue
        elif os.path.exists(result_json_path):
            logger.info("Overwriting %s", full_instance_id)
        else:
            logger.info("Running %s", full_instance_id)

        # Check if retrying failed task
        if args.retry_failed and os.path.exists(result_json_path):
            with open(result_json_path, "r") as f:
                result = json.load(f)
                if result.get("finished") and "FAIL" not in str(result.get("result", "")) and "error" not in str(result.get("result", "")).lower():
                    logger.info("Skipping completed task %s", full_instance_id)
                    continue
            logger.info("Retrying failed task %s", full_instance_id)

        # Clean output directory if exists
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
            logger.info("Removed existing %s", output_dir)

        os.makedirs(output_dir, exist_ok=True)

        # Configure environment
        env_config = {
            "image_name": "spider_agent-image",
            "init_args": {
                "name": experiment_id + "-" + instance_id,
                "work_dir": "/workspace",
            }
        }

        source_data_dir = os.path.dirname(args.test_path)
        task_config['config'] = [{
            "type": "copy_all_subfiles",
            "parameters": {
                "dirs": [os.path.join(source_data_dir, instance_id)]
            }
        }]

        # Create environment
        env = Spider_Agent_Env(
            env_config=env_config,
            task_config=task_config,
            cache_dir="./cache",
            mnt_dir=output_dir
        )

        # Create and configure agent
        agent = DeepSeekDSPyAgent(
            model=args.model,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            max_memory_length=args.max_memory_length,
            max_steps=args.max_steps,
            use_plan=args.plan,
            use_gepa=args.use_gepa,
            gepa_iterations=args.gepa_iterations
        )

        agent.set_env_and_task(env)

        # Run agent
        logger.info('Task input: ' + task_config['question'])
        done, result_output = agent.run()
        trajectory = agent.get_trajectory()

        # Save results
        os.makedirs(os.path.join(output_dir, "spider"), exist_ok=True)
        result_files = env.post_process()
        spider_result = {
            "finished": done,
            "steps": len(trajectory["trajectory"]),
            "result": result_output,
            "result_files": result_files,
            **trajectory
        }

        with open(os.path.join(output_dir, "spider/result.json"), "w") as f:
            json.dump(spider_result, f, indent=2)

        # Clean up SQLite files for local tasks
        if task_type == 'local':
            sqlite_files = glob.glob(os.path.join(output_dir, '*.sqlite')) + glob.glob(os.path.join(output_dir, '*.duckdb'))
            for file_path in sqlite_files:
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")

        logger.info("Finished %s", full_instance_id)
        env.close()


if __name__ == '__main__':
    args = config()
    test(args)