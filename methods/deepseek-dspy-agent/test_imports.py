#!/usr/bin/env python3
"""Test imports for deepseek-dspy-agent."""

import sys
import os

# Add spider-agent-lite to path
current_dir = os.path.dirname(os.path.abspath(__file__))
spider_agent_lite_path = os.path.join(current_dir, "../spider-agent-lite")
if spider_agent_lite_path not in sys.path:
    sys.path.insert(0, spider_agent_lite_path)

print("Testing imports...")

try:
    import dspy
    print("✓ dspy imported")
except ImportError as e:
    print(f"✗ dspy import failed: {e}")

try:
    from openai import OpenAI
    print("✓ openai imported")
except ImportError as e:
    print(f"✗ openai import failed: {e}")

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
except ImportError as e:
    print(f"✗ DeepSeekDSPyAgent import failed: {e}")

print("\nAll imports completed.")