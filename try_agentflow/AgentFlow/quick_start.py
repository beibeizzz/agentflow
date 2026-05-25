from pathlib import Path
import sys
import types

# AgentFlow's solver package expects the inner runtime package to be importable
# as top-level "agentflow". Bootstrap that package explicitly for this script.
agentflow_core = Path(__file__).resolve().parent / "agentflow" / "agentflow"
agentflow_pkg = types.ModuleType("agentflow")
agentflow_pkg.__path__ = [str(agentflow_core)]
agentflow_pkg.__file__ = str(agentflow_core / "__init__.py")
sys.modules["agentflow"] = agentflow_pkg

from agentflow.solver import construct_solver

llm_engine_name = "vllm-Qwen3-0.6B-Instruct"
base_url = "http://localhost:8000/v1"

solver = construct_solver(
    llm_engine_name=llm_engine_name,
    base_url=base_url,
    enabled_tools=["Base_Generator_Tool"],
    tool_engine=["self"],
    model_engine=["trainable", "trainable", "trainable", "trainable"],
    output_types="direct",
)

output = solver.solve("What is the capital of France?")
print(output["direct_output"])
