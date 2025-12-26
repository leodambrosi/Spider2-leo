import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

# Add spider-agent-lite to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
spider_agent_lite_path = os.path.join(current_dir, "../spider-agent-lite")
if spider_agent_lite_path not in sys.path:
    sys.path.insert(0, spider_agent_lite_path)

import dspy
from openai import OpenAI

try:
    from gepa_optimizer import GEPA
    GEPA_AVAILABLE = True
except ImportError:
    GEPA_AVAILABLE = False
    # Fallback to BootstrapFewShot
    from dspy.teleprompt import BootstrapFewShot
from spider_agent.envs.spider_agent import Spider_Agent_Env
from spider_agent.agent.action import Action, Bash, Terminate, CreateFile, EditFile, LOCAL_DB_SQL, BIGQUERY_EXEC_SQL, SNOWFLAKE_EXEC_SQL, BQ_GET_TABLES, BQ_GET_TABLE_INFO, BQ_SAMPLE_ROWS
from spider_agent.agent.prompts import BIGQUERY_SYSTEM, LOCAL_SYSTEM, DBT_SYSTEM, SNOWFLAKE_SYSTEM, REFERENCE_PLAN_SYSTEM
from dataclasses import dataclass

@dataclass
class TaskConfig:
    action_classes: list
    system_template: str


# Task type configurations mapping to TaskConfig instances
TASK_CONFIGS = {
    'Bigquery': TaskConfig(
        action_classes=[Bash, Terminate, BIGQUERY_EXEC_SQL, CreateFile, EditFile],
        system_template=BIGQUERY_SYSTEM
    ),
    'Snowflake': TaskConfig(
        action_classes=[Bash, Terminate, SNOWFLAKE_EXEC_SQL, CreateFile, EditFile],
        system_template=SNOWFLAKE_SYSTEM
    ),
    'Local': TaskConfig(
        action_classes=[Bash, Terminate, CreateFile, EditFile, LOCAL_DB_SQL],
        system_template=LOCAL_SYSTEM
    ),
    'DBT': TaskConfig(
        action_classes=[Bash, Terminate, CreateFile, EditFile, LOCAL_DB_SQL],
        system_template=DBT_SYSTEM
    ),
}

# Custom exceptions
class DeepSeekAgentError(Exception):
    """Base exception for DeepSeek-DSPy agent errors."""
    pass

class ConfigurationError(DeepSeekAgentError):
    """Configuration-related errors."""
    pass

class AgentNotConfiguredError(ConfigurationError):
    """Agent not properly configured before execution."""
    pass

class ExecutionError(DeepSeekAgentError):
    """Errors during agent execution."""
    pass

class RepeatedActionError(ExecutionError):
    """Action repeated multiple times."""
    pass

class ActionParseError(ExecutionError):
    """Failed to parse action text."""
    pass

class StepLimitExceededError(ExecutionError):
    """Maximum steps exceeded."""
    pass

class APIError(DeepSeekAgentError):
    """API-related errors."""
    pass


class ActionParser:
    """Parses action text into Action objects."""

    def __init__(self, available_action_classes: List):
        self.available_action_classes = available_action_classes

    def parse(self, action_text: str):
        """Parse action text into Action object, prioritizing JSON."""
        if not action_text:
            return None

        # 1. Try to parse as a direct JSON object
        try:
            # Handle potential markdown code blocks first
            json_text = action_text
            if "```" in action_text:
                matches = re.findall(r"```(?:json)?\n?([\s\S]*?)```", action_text)
                if matches:
                    json_text = matches[-1].strip()
            
            data = json.loads(json_text)
            if isinstance(data, dict) and "action" in data:
                action_candidate = data["action"]
                # Try to parse the action string from JSON
                for action_cls in self.available_action_classes:
                    action = action_cls.parse_action_from_text(action_candidate)
                    if action is not None:
                        return action
        except (json.JSONDecodeError, TypeError):
            pass

        # 2. Fallback: Search for action patterns in the raw text
        # This handles cases where the model ignores the JSON instruction 
        # but still outputs a valid ActionName(...) string.
        for action_cls in self.available_action_classes:
            action = action_cls.parse_action_from_text(action_text)
            if action is not None:
                return action

        return None


class MemoryManager:
    """Manages agent memory and history."""

    def __init__(self, max_memory_length: int = 10):
        self.max_memory_length = max_memory_length
        self.history = []  # list of dicts with observation, thought, action
        self.observations = []
        self.thoughts = []
        self.responses = []
        self.actions = []

    def add_step(
        self,
        observation: str,
        thought: str,
        response: str,
        action
    ):
        """Add a step to memory."""
        # Add to history dict
        self.history.append({
            "observation": observation,
            "thought": thought,
            "action": response  # Note: response is the action text
        })

        # Add to individual lists
        self.observations.append(observation)
        self.thoughts.append(thought)
        self.responses.append(response)
        self.actions.append(action)

        # Trim history if exceeds max length
        if len(self.history) > self.max_memory_length:
            self.history = self.history[-self.max_memory_length:]
            self.observations = self.observations[-self.max_memory_length:]
            self.thoughts = self.thoughts[-self.max_memory_length:]
            self.responses = self.responses[-self.max_memory_length:]
            self.actions = self.actions[-self.max_memory_length:]

    def get_recent_history(self, max_length: int = None) -> str:
        """Get formatted recent history string."""
        if not self.history:
            return ""

        length = max_length or self.max_memory_length
        recent = self.history[-length:]

        return "\n".join([
            f"Step {i}: {h['thought']}\nAction: {h['action']}\nObservation: {h['observation']}"
            for i, h in enumerate(recent, start=1)
        ])

    def clear(self):
        """Clear all memory."""
        self.history.clear()
        self.observations.clear()
        self.thoughts.clear()
        self.responses.clear()
        self.actions.clear()


logger = logging.getLogger("deepseek_dspy_agent")


class DeepSeekClient:
    """Client for DeepSeek Reasoner API communication."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-reasoner",
        base_url: str = "https://api.deepseek.com",
        **kwargs
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.kwargs = {
            "max_tokens": 2000,
            "temperature": 1.0,
            "top_p": 0.9,
            **kwargs
        }

    def chat_completion(self, messages: list, **override_kwargs) -> dict:
        """Make chat completion request to DeepSeek API."""
        combined_kwargs = {**self.kwargs, **override_kwargs}
        
        # Handle JSON mode if requested
        response_format = combined_kwargs.pop('response_format', None)
        extra_body = combined_kwargs.pop('extra_body', None)

        try:
            logger.debug(f"DeepSeek API request: model={self.model}, messages={messages}")
            
            # Construct parameters for the OpenAI client
            params = {
                "model": self.model,
                "messages": messages,
                **combined_kwargs
            }
            if response_format:
                params["response_format"] = response_format
            if extra_body:
                params["extra_body"] = extra_body

            response = self.client.chat.completions.create(**params)

            message = response.choices[0].message
            content = message.content

            # Extract reasoning content for DeepSeek R1 models
            reasoning_content = None
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                reasoning_content = message.reasoning_content
                logger.debug(f"DeepSeek R1 reasoning content (first 1000 chars): {reasoning_content[:1000] if reasoning_content else 'EMPTY'}")

            logger.debug(f"DeepSeek API response (first 1000 chars): {content[:1000] if content else 'EMPTY'}")

            return {
                'content': content,
                'reasoning_content': reasoning_content
            }
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise APIError(f"DeepSeek API request failed: {e}") from e


class DeepSeekReasonerLM(dspy.LM):
    """DSPy LM wrapper for DeepSeek Reasoner model."""

    def __init__(self, api_key: str, model: str = "deepseek-reasoner", **kwargs):
        super().__init__(model)
        self.api_key = api_key
        self.model = model
        self.client = DeepSeekClient(api_key=api_key, model=model, **kwargs)
        self.system_message = "You are a helpful assistant."
        self.last_reasoning_content = None

    def set_system_message(self, system_message: str):
        """Update the system message used for all requests."""
        self.system_message = system_message

    def _prepare_messages(self, messages):
        """Inject or prepend the agent's system message to messages."""
        if not self.system_message:
            return messages

        # Check if first message is a system message
        if messages and messages[0]['role'] == 'system':
            existing_content = messages[0]['content']
            # If our system message is already present, skip duplication
            if self.system_message in existing_content:
                return messages
            # Combine our system message with existing system message
            combined = f"{self.system_message}\n\n---\n\n{existing_content}"
            messages[0]['content'] = combined
        else:
            # Insert our system message as first message
            messages.insert(0, {'role': 'system', 'content': self.system_message})
        return messages

    def _extract_thought_action(self, text: str) -> tuple[str, str]:
        """Extract thought and action from reasoning text."""
        # Look for Thought: ... Action: ... patterns
        thought = None
        action = None

        import re
        thought_match = re.search(r'Thought:\s*(.*?)(?=\nAction:|\n|$)', text, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()

        action_match = re.search(r'Action:\s*(.*?)(?=\n|$)', text, re.DOTALL | re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()

        return thought, action

    def basic_request(self, prompt: str = None, **kwargs):
        """Make request to DeepSeek API."""
        # Check if messages are provided directly (DSPy 2.x style)
        if 'messages' in kwargs:
            messages = kwargs.pop('messages')
        else:
            # Construct messages from prompt (DSPy 3.x style)
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt or ''}
            ]

        # Ensure our system message is included
        messages = self._prepare_messages(messages)

        try:
            logger.debug(f"DeepSeek API request: messages={messages}")
            response = self.client.chat_completion(messages, **kwargs)

            content = response['content']
            reasoning_content = response['reasoning_content']
            logger.debug(f"DeepSeek R1 reasoning content (first 1000 chars): {reasoning_content[:1000] if reasoning_content else 'EMPTY'}")

            # Store reasoning content for agent to use
            self.last_reasoning_content = reasoning_content

            # If content is empty, try to construct JSON from reasoning content
            if not content or content.strip() == '':
                thought, action = self._extract_thought_action(reasoning_content or '')
                if thought is not None and action is not None:
                    # Construct JSON string
                    import json
                    content = json.dumps({'thought': thought, 'action': action})
                else:
                    # Fallback: use reasoning content as thought, default action
                    thought = reasoning_content or 'No thought generated'
                    action = "Bash(code='ls -la')"
                    import json
                    content = json.dumps({'thought': thought, 'action': action})

            logger.debug(f"DeepSeek API response (first 1000 chars): {content[:1000] if content else 'EMPTY'}")
            return content
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise

    def __call__(self, prompt: str = None, **kwargs):
        # Handle both positional and keyword arguments
        if prompt is None:
            prompt = kwargs.pop('prompt', '')
        return self.basic_request(prompt, **kwargs)


class ThoughtActionSignature(dspy.Signature):
    """
    Reason about the next step and provide a structured response.
    
    You must output your response as a JSON object with two keys:
    1. "thought": Your reasoning about what to do next.
    2. "action": The action to take, formatted exactly as ActionName(parameter='value').
    
    Example Output:
    {
        "thought": "I need to check the tables in the database.",
        "action": "LOCAL_DB_SQL(query='SELECT name FROM sqlite_master')"
    }
    """
    observation = dspy.InputField(desc="Current observation from environment")
    history = dspy.InputField(desc="Recent interaction history")
    instruction = dspy.InputField(desc="Original task instruction")
    thought = dspy.OutputField(desc="Reasoning about what to do next")
    action = dspy.OutputField(desc="Action to take (in proper format)")


class DeepSeekDSPyAgent:
    """Agent using DeepSeek Reasoner with DSPy optimization."""

    def __init__(
        self,
        model: str = "deepseek-reasoner",
        api_key: str = None,
        max_tokens: int = 2000,
        top_p: float = 0.9,
        temperature: float = 1.0,
        max_memory_length: int = 10,
        max_steps: int = 15,
        use_plan: bool = False,
        use_gepa: bool = False,
        gepa_iterations: int = 10,
        thinking: bool = False,
        json_mode: bool = False
    ):
        # Set up DeepSeek LM
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")

        # Configure extra parameters for thinking and JSON mode
        extra_kwargs = {}
        if thinking:
            extra_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        if json_mode:
            extra_kwargs["response_format"] = {"type": "json_object"}

        self.lm = DeepSeekReasonerLM(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **extra_kwargs
        )

        # Configure DSPy
        dspy.configure(lm=self.lm)

        # Create DSPy module
        self.module = dspy.Predict(ThoughtActionSignature)

        # GEPA optimizer
        self.use_gepa = use_gepa
        self.gepa_iterations = gepa_iterations
        self.optimized_module = None
        if use_gepa:
            logger.warning(
                "GEPA optimization is experimental with basic genetic operations implemented. "
                "It performs text mutations on signatures but may provide limited optimization. "
                "Consider using --no_gepa for baseline performance comparison."
            )
            if GEPA_AVAILABLE:
                # Use our GEPA optimizer
                self.teleprompter = GEPA(
                    metric=self._evaluate_response,
                    population_size=20,
                    generations=gepa_iterations
                )
            else:
                # Fallback to BootstrapFewShot
                self.teleprompter = BootstrapFewShot(
                    metric=self._evaluate_response,
                    max_bootstrapped_demos=3,
                    max_labeled_demos=5
                )

        # Agent state
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.max_memory_length = max_memory_length
        self.max_steps = max_steps
        self.use_plan = use_plan

        self.memory_manager = MemoryManager(max_memory_length=max_memory_length)
        self.system_message = ""
        self.env = None
        self.instruction = ""

        self._AVAILABLE_ACTION_CLASSES = []
        self.action_parser = ActionParser([])

    def _evaluate_response(self, example, pred, trace=None):
        """Evaluate response for optimization."""
        # Check if action parsed successfully
        try:
            action = self.action_parser.parse(pred.action)
            if action is None:
                return 0.0
            
            score = 1.0
            
            # Bonus for appropriate action types based on task type
            if self.env:
                task_type = self.env.task_config.get('type', 'Local')
                action_type = type(action).__name__
                
                if task_type == 'Bigquery' and 'BIGQUERY' in action_type:
                    score += 0.5
                elif task_type == 'Snowflake' and 'SNOWFLAKE' in action_type:
                    score += 0.5
                elif task_type in ['Local', 'DBT'] and 'SQL' in action_type:
                    score += 0.5
            
            # Penalize excessively long actions (likely hallucinations or messy output)
            if len(pred.action) > 1000:
                score -= 0.5
                
            return max(0.0, score)
        except Exception:
            return 0.0

    def _create_training_examples(self):
        """Create task-specific training examples for optimization."""
        if not self.env:
            # Fallback to generic examples
            return self._create_generic_examples()

        task_type = self.env.task_config.get('type', 'Local')

        if task_type == 'Bigquery':
            return self._create_bigquery_examples()
        elif task_type == 'Snowflake':
            return self._create_snowflake_examples()
        elif task_type == 'Local':
            return self._create_local_examples()
        elif task_type == 'DBT':
            return self._create_dbt_examples()
        else:
            return self._create_generic_examples()

    def _create_generic_examples(self):
        """Create generic training examples (fallback)."""
        return [
            dspy.Example(
                observation="You are in the folder now.",
                history="",
                instruction="List files in current directory",
                thought="I should list files to see what's available.",
                action="Bash(command='ls -la')"
            ).with_inputs("observation", "history", "instruction"),
            dspy.Example(
                observation="File listing shows: README.md, data.csv",
                history="Step 1: I should list files to see what's available.\nAction: Bash(command='ls -la')\nObservation: You are in the folder now.",
                instruction="Count the number of files",
                thought="I can use wc -l to count files, but need to exclude directories. Use ls -1 | wc -l.",
                action="Bash(command='ls -1 | wc -l')"
            ).with_inputs("observation", "history", "instruction"),
            dspy.Example(
                observation="File count: 5",
                history="Previous steps...",
                instruction="Create a new file called test.txt",
                thought="I should use touch command to create a new file.",
                action="Bash(command='touch test.txt')"
            ).with_inputs("observation", "history", "instruction"),
        ]

    def _create_bigquery_examples(self):
        """Create BigQuery-specific training examples."""
        return [
            dspy.Example(
                observation="Connected to BigQuery project. Use BIGQUERY_EXEC_SQL to query tables.",
                history="",
                instruction="Show available tables in the dataset",
                thought="I need to query INFORMATION_SCHEMA.TABLES to see available tables.",
                action="BIGQUERY_EXEC_SQL(query='SELECT table_name FROM `project.dataset.INFORMATION_SCHEMA.TABLES` LIMIT 10')"
            ).with_inputs("observation", "history", "instruction"),
            dspy.Example(
                observation="Tables list: customers, orders, products",
                history="Step 1: Show available tables...",
                instruction="Count rows in customers table",
                thought="I should use SELECT COUNT(*) FROM customers.",
                action="BIGQUERY_EXEC_SQL(query='SELECT COUNT(*) FROM `project.dataset.customers`')"
            ).with_inputs("observation", "history", "instruction"),
            dspy.Example(
                observation="Count: 1500 rows",
                history="Previous steps...",
                instruction="Get schema of orders table",
                thought="Query INFORMATION_SCHEMA.COLUMNS for the orders table.",
                action="BIGQUERY_EXEC_SQL(query='SELECT column_name, data_type FROM `project.dataset.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = \"orders\"')"
            ).with_inputs("observation", "history", "instruction"),
        ]

    def _create_snowflake_examples(self):
        """Create Snowflake-specific training examples."""
        return [
            dspy.Example(
                observation="Connected to Snowflake database. Use SNOWFLAKE_EXEC_SQL to query.",
                history="",
                instruction="List tables in current schema",
                thought="I need to query INFORMATION_SCHEMA.TABLES.",
                action="SNOWFLAKE_EXEC_SQL(query='SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = CURRENT_SCHEMA()')"
            ).with_inputs("observation", "history", "instruction"),
            dspy.Example(
                observation="Tables: users, transactions, products",
                history="Step 1: List tables...",
                instruction="Count active users",
                thought="SELECT COUNT(*) FROM users WHERE active = true.",
                action="SNOWFLAKE_EXEC_SQL(query='SELECT COUNT(*) FROM users WHERE active = TRUE')"
            ).with_inputs("observation", "history", "instruction"),
        ]

    def _create_local_examples(self):
        """Create local SQLite-specific training examples."""
        return [
            dspy.Example(
                observation="Connected to local SQLite database. Use LOCAL_DB_SQL to query.",
                history="",
                instruction="List all tables",
                thought="Query sqlite_master table for table names.",
                action="LOCAL_DB_SQL(query='SELECT name FROM sqlite_master WHERE type=\"table\"')"
            ).with_inputs("observation", "history", "instruction"),
            dspy.Example(
                observation="Tables: employees, departments, salaries",
                history="Step 1: List tables...",
                instruction="Count employees in Engineering department",
                thought="Join employees and departments tables.",
                action="LOCAL_DB_SQL(query='SELECT COUNT(*) FROM employees e JOIN departments d ON e.dept_id = d.id WHERE d.name = \"Engineering\"')"
            ).with_inputs("observation", "history", "instruction"),
        ]

    def _create_dbt_examples(self):
        """Create DBT-specific training examples (similar to local)."""
        return self._create_local_examples()  # DBT uses local SQL

    def set_env_and_task(self, env: Spider_Agent_Env):
        """Set environment and task configuration."""
        self.env = env
        self.memory_manager.clear()
        self.instruction = self.env.task_config['question']

        # Configure action space based on task type
        task_type = self.env.task_config['type']
        if task_type not in TASK_CONFIGS:
            raise ConfigurationError(f"Unsupported task type: {task_type}")

        config = TASK_CONFIGS[task_type]
        self._AVAILABLE_ACTION_CLASSES = config.action_classes
        action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
        self.system_message = config.system_template.format(
            work_dir="/workspace",
            action_space=action_space,
            task=self.instruction,
            max_steps=self.max_steps
        )

        self.action_parser = ActionParser(self._AVAILABLE_ACTION_CLASSES)

        if self.use_plan and hasattr(self, 'reference_plan'):
            self.system_message += REFERENCE_PLAN_SYSTEM.format(plan=self.reference_plan)

        # Update LM with system message
        self.lm.set_system_message(self.system_message)

        # Initialize DSPy module with system message
        self.module = dspy.Predict(ThoughtActionSignature)

        # Optimize module if GEPA is enabled
        if self.use_gepa and self.optimized_module is None:
            self._optimize_module()

    def _optimize_module(self):
        """Optimize DSPy module using GEPA or fallback optimizer."""
        # Create task-specific training examples
        training_examples = self._create_training_examples()
        logger.info(f"Created {len(training_examples)} training examples for optimization")

        # Use teleprompter to optimize
        try:
            self.optimized_module = self.teleprompter.compile(
                self.module,
                trainset=training_examples
            )
            logger.info(f"Module optimized with {'GEPA' if GEPA_AVAILABLE else 'BootstrapFewShot'}")
        except Exception as e:
            logger.warning(f"Optimization failed: {e}. Using uncompiled module.")
            self.optimized_module = self.module

    def predict(self, obs: str = None) -> Tuple[str, Optional[Action]]:
        """Predict next action using DSPy module."""
        # Prepare history context
        history_context = self.memory_manager.get_recent_history()

        # Use optimized module if available
        module_to_use = self.optimized_module if self.optimized_module else self.module

        # Generate thought and action
        try:
            result = module_to_use(
                observation=obs or "",
                history=history_context,
                instruction=self.instruction
            )
        except ValueError as e:
            logger.warning(f"DSPy module failed with ValueError: {e}. Falling back to default.")
            # Create a dummy result with defaults
            class Result:
                pass
            result = Result()
            result.thought = "Fallback: DSPy module failed."
            result.action = "Bash(code='ls -la')"

        # Use reasoning content if available (DeepSeek R1)
        if self.lm.last_reasoning_content:
            thought = self.lm.last_reasoning_content
            # Clear it after use to avoid accidental reuse
            self.lm.last_reasoning_content = None
            logger.debug("Using DeepSeek R1 reasoning content for thought")
        else:
            thought = result.thought

        action_text = result.action

        # Parse action
        action = self.action_parser.parse(action_text)

        # Update memory
        self.memory_manager.add_step(obs, thought, action_text, action)

        logger.info(f"Observation: {obs}")
        logger.info(f"Thought: {thought}")
        logger.info(f"Action: {action_text}")

        return action_text, action


    def run(self) -> Tuple[bool, str]:
        """Run agent on current task."""
        if self.env is None:
            raise AgentNotConfiguredError(
                "Environment not set. Call set_env_and_task() before run()."
            )

        result = ""
        done = False
        step_idx = 0
        obs = "You are in the folder now."
        retry_count = 0
        last_action = None
        repeat_action = False

        while not done and step_idx < self.max_steps:
            _, action = self.predict(obs)

            if action is None:
                logger.warning("Failed to parse action, retrying...")
                retry_count += 1
                if retry_count > 3:
                    logger.error("Max retries exceeded")
                    break
                obs = "Failed to parse action from your response. Please provide a valid action."
            else:
                logger.info(f"Step {step_idx + 1}: {action}")

                # Check for repeated actions
                if last_action is not None and str(last_action) == str(action):
                    if repeat_action:
                        return False, "ERROR: Repeated action"
                    else:
                        obs = "The action is the same as the last one. Please provide a DIFFERENT action."
                        repeat_action = True
                else:
                    # Execute action in environment
                    obs, done = self.env.step(action)
                    last_action = action
                    repeat_action = False

            if done:
                if isinstance(action, Terminate):
                    result = action.output
                logger.info("Task completed")
                break

            step_idx += 1

        return done, result

    def get_trajectory(self) -> Dict:
        """Get full trajectory of agent execution."""
        trajectory = []
        for i in range(len(self.memory_manager.observations)):
            trajectory.append({
                "observation": self.memory_manager.observations[i],
                "thought": self.memory_manager.thoughts[i],
                "action": str(self.memory_manager.actions[i]) if self.memory_manager.actions[i] else None,
                "response": self.memory_manager.responses[i]
            })

        return {
            "Task": self.instruction,
            "system_message": self.system_message,
            "trajectory": trajectory
        }