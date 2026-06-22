from pathlib import Path
import sys
import types


PROJECT_DIR = Path(__file__).resolve().parents[2]
AGENTFLOW_CORE = PROJECT_DIR / "agentflow" / "agentflow"
agentflow_pkg = types.ModuleType("agentflow")
agentflow_pkg.__path__ = [str(AGENTFLOW_CORE)]
agentflow_pkg.__file__ = str(AGENTFLOW_CORE / "__init__.py")
sys.modules["agentflow"] = agentflow_pkg
