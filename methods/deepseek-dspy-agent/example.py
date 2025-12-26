#!/usr/bin/env python3
"""
Example usage of DeepSeekDSPyAgent with a mock environment.

This demonstrates how to use the agent without requiring the full
Spider 2.0 environment setup.
"""

import os
import sys
from typing import Tuple, Any

# Add spider-agent-lite to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
spider_agent_lite_path = os.path.join(current_dir, "../spider-agent-lite")
if spider_agent_lite_path not in sys.path:
    sys.path.insert(0, spider_agent_lite_path)

# Mock environment for demonstration
class MockEnv:
    """Mock environment for testing."""

    def __init__(self):
        self.task_config = {
            'type': 'Local',
            'question': 'List files and count them',
            'instance_id': 'mock_001'
        }
        self.step_count = 0

    def step(self, action):
        """Mock step function."""
        self.step_count += 1
        action_str = str(action)

        # Simulate environment responses
        if 'ls' in action_str:
            return "Files: README.md, example.py, test.txt", False
        elif 'wc' in action_str:
            return "Count: 3 files", False
        elif 'touch' in action_str:
            return "File created", False
        elif 'Terminate' in action_str:
            return "Task completed", True
        else:
            return f"Unknown action: {action_str}", False

    def post_process(self):
        """Mock post-process."""
        return ["mock_result.csv"]


# Mock action classes for testing
class MockAction:
    """Base mock action."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __str__(self):
        return f"{self.__class__.__name__}({self.kwargs})"

    @classmethod
    def parse_action_from_text(cls, text):
        """Simple parser for mock actions."""
        if cls.__name__ in text:
            # Very simple parsing - just create action
            return cls()
        return None


class Bash(MockAction):
    pass


class Terminate(MockAction):
    def __init__(self, output=""):
        super().__init__(output=output)
        self.output = output


def mock_imports():
    """Mock imports for demonstration."""
    # This would be replaced with real imports when dependencies are installed
    print("Note: This example uses mock classes.")
    print("Install dependencies with: pip install -r requirements.txt")
    print("Set DEEPSEEK_API_KEY environment variable.")
    return MockAction, Bash, Terminate


def main():
    """Run example with mock environment."""
    print("=== DeepSeek DSPy Agent Example ===\n")

    try:
        from deepseek_agent import DeepSeekDSPyAgent
        print("✓ Successfully imported DeepSeekDSPyAgent")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("\nTrying with mock imports...")
        # Create a simplified version for demonstration
        import types

        # Create a mock agent class
        class MockAgent:
            def __init__(self, **kwargs):
                self.model = kwargs.get('model', 'deepseek-reasoner')
                self.max_steps = kwargs.get('max_steps', 5)
                self.thoughts = []
                self.actions = []
                self.observations = []

            def set_env_and_task(self, env):
                self.env = env
                self.instruction = env.task_config['question']
                print(f"Task: {self.instruction}")

            def run(self):
                print("Mock agent running...")
                print("(In real usage, this would call DeepSeek Reasoner via DSPy)")
                return True, "Mock result"

        agent_class = MockAgent
        print("Using mock agent class for demonstration.\n")

    # Create mock environment
    env = MockEnv()

    # Create agent (mock or real)
    if 'agent_class' in locals() and agent_class.__name__ == 'MockAgent':
        agent = agent_class(model='deepseek-reasoner', max_steps=3)
    else:
        # Real agent
        agent = DeepSeekDSPyAgent(
            model='deepseek-reasoner',
            max_steps=5,
            use_gepa=False  # Disable GEPA for quick example
        )

    # Set environment
    agent.set_env_and_task(env)

    # Run agent
    print("Starting agent execution...")
    done, result = agent.run()

    print(f"\nExecution completed: {done}")
    print(f"Result: {result}")

    print("\n=== Example Complete ===")
    print("\nTo run the full agent:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Set DEEPSEEK_API_KEY environment variable")
    print("3. Run: python run.py --model deepseek-reasoner --suffix test")
    print("4. For GEPA optimization: python run.py --use_gepa --gepa_iterations 10")


if __name__ == '__main__':
    main()