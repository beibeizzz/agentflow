"""Export default single-entry schemas; actual run prompts reflect configured tools."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentflow_rl.runtime.contracts import ToolName
from agentflow_rl.tools.catalog import CATALOG_REVISION, tool_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/inspection/tool_catalog.default.json'))
    args = parser.parse_args()
    payload = {'revision': CATALOG_REVISION, 'configuration': 'defaults',
               'planner': [tool_spec(name).planner_view() for name in ToolName],
               'executor': [tool_spec(name).executor_view() for name in ToolName],
               'validation_schemas': [tool_spec(name).view(executor=True) for name in ToolName]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
