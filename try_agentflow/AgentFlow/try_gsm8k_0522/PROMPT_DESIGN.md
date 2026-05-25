# Calculator-Only Prompt Design

The prompt is now split by AgentFlow role. The GSM8K dataset query no longer contains the full `Planner` / `Executor` / `Verifier` / `Generator` instructions.

## Dataset Query

File: `prepare_gsm8k_json.py`

Recipient: every AgentFlow role sees this text as the user problem.

Purpose:
- Provide the GSM8K problem.
- Avoid the incorrect pre-tool template `<Calculator> expression = result <Calculator>`.

## Planner Analysis

File: `agentflow/agentflow/models/planner.py`
Method: `Planner.analyze_query`

Recipient: Planner analysis LLM call.

Purpose:
- Extract quantities, units, and the unknown.
- Build an arithmetic plan.
- Mention `Calculator_Tool` as the only arithmetic tool.
- Do not call tools or invent calculator results.

## Planner Next Step

File: `agentflow/agentflow/models/planner.py`
Method: `Planner.generate_next_step`

Recipient: Planner action-prediction LLM call.

Purpose:
- Choose exactly one next calculator expression.
- Put a raw numeric expression in context.
- Use `Tool Name: Calculator_Tool` exactly.
- Do not include the expected result before the tool is called.

## Executor

File: `agentflow/agentflow/models/executor.py`
Method: `Executor.generate_tool_command`

Recipient: Executor tool-command LLM call.

Purpose:
- Convert the planner's raw expression into Python tool code.
- Generate `execution = tool.execute(expression="...")`.
- Never use `query=`.
- Never pass natural language or the full problem into the calculator.

## Verifier

File: `agentflow/agentflow/models/verifier.py`
Method: `Verifier.verificate_context`

Recipient: Verifier LLM call.

Purpose:
- Check whether successful calculator results are sufficient.
- Interpret each memory action as `command` = arithmetic expression and `result` = numeric calculator output.
- Return `STOP` only when the arithmetic chain can answer the problem.
- Return `CONTINUE` when a missing command, wrong tool, empty result, or calculator error remains.

## Generator

File: `agentflow/agentflow/models/planner.py`
Method: `Planner.generate_direct_output`

Recipient: final answer LLM call.

Purpose:
- Use memory actions where `command` contains the expression and `result` contains only the numeric value.
- Return the final numeric result only.

The `Generator` answer should contain only the final numeric answer: no units, prose, markdown, or currency symbols.
