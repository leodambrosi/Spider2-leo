#!/usr/bin/env python3
"""Test script to run the DeepSeek DSPy agent with proper environment setup."""

import os
import sys
from pathlib import Path

# Load environment variables from .env file in parent directory
try:
    from dotenv import load_dotenv
    # Go up two levels from current file to project root
    env_path = Path(__file__).parent.parent.parent / '.env'
    print(f"Loading environment from: {env_path}")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print("✓ Environment loaded from .env file")
    else:
        print(f"✗ .env file not found at {env_path}")
except ImportError:
    print("✗ python-dotenv not installed")

# Check for DEEPSEEK_API_KEY
if 'DEEPSEEK_API_KEY' in os.environ:
    print(f"✓ DEEPSEEK_API_KEY found (length: {len(os.environ['DEEPSEEK_API_KEY'])})")
else:
    print("✗ DEEPSEEK_API_KEY not found in environment")
    sys.exit(1)

# Add spider-agent-lite to path
current_dir = os.path.dirname(os.path.abspath(__file__))
spider_agent_lite_path = os.path.join(current_dir, "../spider-agent-lite")
if spider_agent_lite_path not in sys.path:
    sys.path.insert(0, spider_agent_lite_path)

print("\nTesting imports...")

try:
    import dspy
    print("✓ dspy imported")
except ImportError as e:
    print(f"✗ dspy import failed: {e}")

try:
    from openai import OpenAI
    print("✓ openai imported")

    # Test OpenAI client initialization
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    print("✓ OpenAI client initialized")
except ImportError as e:
    print(f"✗ openai import failed: {e}")
except Exception as e:
    print(f"✗ OpenAI client test failed: {e}")

try:
    from spider_agent.envs.spider_agent import Spider_Agent_Env
    print("✓ Spider_Agent_Env imported")
except ImportError as e:
    print(f"✗ Spider_Agent_Env import failed: {e}")

try:
    from spider_agent.agent.action import Action
    print("✓ Action imported")
except ImportError as e:
    print(f"✗ Action import failed: {e}")

try:
    from deepseek_agent import DeepSeekDSPyAgent
    print("✓ DeepSeekDSPyAgent imported")

    # Try to create an instance
    agent = DeepSeekDSPyAgent(
        model='deepseek-reasoner',
        max_steps=5,
        use_gepa=False
    )
    print("✓ DeepSeekDSPyAgent instance created")
except ImportError as e:
    print(f"✗ DeepSeekDSPyAgent import failed: {e}")
except Exception as e:
    print(f"✗ DeepSeekDSPyAgent creation failed: {e}")

print("\n✓ All tests completed")