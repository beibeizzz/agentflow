# AgentFlow GSM8K Prompt Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the calculator-only AgentFlow prompts so the existing Planner -> Executor -> Tool -> Memory -> Verifier loop works reliably for GSM8K with `Calculator_Tool`.

**Architecture:** Keep the current continuous planning workflow. Planner performs `analyze_query`, then each loop calls `generate_next_step`; Executor converts the Planner sub-goal into `execution = tool.execute(expression="...")`; `execute_tool_command()` executes matched `tool.execute(...)` commands; Verifier decides `STOP` or `CONTINUE`; Generator produces the final numeric answer from Memory.

**Tech Stack:** Python, AgentFlow local package, vLLM OpenAI-compatible chat endpoint, custom `Calculator_Tool`, `unittest`.

---

## File Structure

- Modify `try_agentflow/AgentFlow/agentflow/agentflow/models/planner.py`
  - Update calculator-only prompt in `analyze_query()`.
  - Update calculator-only prompt in `generate_next_step()`.
  - Update calculator-only Generator prompts in `generate_final_output()` and `generate_direct_output()`.
- Modify `try_agentflow/AgentFlow/agentflow/agentflow/models/executor.py`
  - Update calculator-only prompt in `generate_tool_command()`.
- Modify `try_agentflow/AgentFlow/agentflow/agentflow/models/verifier.py`
  - Update calculator-only prompt in `verificate_context()`.
- Modify or add tests under `try_agentflow/AgentFlow/try_gsm8k_0522/tests/`
  - Ensure each prompt explicitly matches the retained workflow contract.
  - Ensure no prompt asks for `<Calculator>...</Calculator>` as an executable format.
  - Ensure Executor prompt requires `execution = tool.execute(expression="...")`.
- Run smoke from `try_agentflow/AgentFlow/try_gsm8k_0522`.

---

### Task 1: Add Prompt Contract Tests

**Files:**
- Modify: `try_agentflow/AgentFlow/try_gsm8k_0522/tests/test_role_prompts.py`
- Test: `try_agentflow/AgentFlow/try_gsm8k_0522/tests/test_role_prompts.py`

- [x] **Step 1: Add tests for the retained workflow contract**

Add assertions to the existing prompt tests, or add new tests if the current test bodies are too narrow:

```python
def test_planner_next_step_prompt_keeps_continuous_planning_contract(self):
    captured = {}

    class FakeLLM:
        def __call__(self, prompt, **kwargs):
            captured["prompt"] = prompt
            return (
                "Justification: next unresolved calculation\n"
                "Context: Expression to calculate: 16 - 3 - 4. Meaning: remaining eggs.\n"
                "Sub-Goal: Calculate 16 - 3 - 4.\n"
                "Tool Name: Calculator_Tool"
            )

    from agentflow.models.memory import Memory
    from agentflow.models.planner import Planner

    planner = Planner.__new__(Planner)
    planner.available_tools = ["Calculator_Tool"]
    planner.toolbox_metadata = {"Calculator_Tool": {"input_types": {"expression": "str"}}}
    planner.is_multimodal = False
    planner.llm_engine = FakeLLM()

    planner.generate_next_step(
        question="Janet has 16 eggs, eats 3, bakes 4. How many remain?",
        image=None,
        query_analysis="Need subtract eaten and baked eggs.",
        memory=Memory(),
        step_count=1,
        max_step_count=3,
        json_data={},
    )

    prompt = captured["prompt"]
    self.assertIn("existing AgentFlow loop", prompt)
    self.assertIn("next single unresolved arithmetic expression", prompt)
    self.assertIn("Tool Name: Calculator_Tool", prompt)
    self.assertIn("Do not include the result", prompt)
    self.assertNotIn("<Calculator>", prompt)
```

```python
def test_executor_prompt_requires_tool_execute_expression_contract(self):
    captured = {}

    class FakeLLM:
        def __call__(self, prompt, **kwargs):
            captured["prompt"] = prompt
            return '```python\nexecution = tool.execute(expression="16 - 3 - 4")\n```'

    from agentflow.models.executor import Executor

    executor = Executor.__new__(Executor)
    executor.llm_generate_tool_command = FakeLLM()

    executor.generate_tool_command(
        question="Janet has 16 eggs, eats 3, bakes 4. How many remain?",
        image=None,
        context="Expression to calculate: 16 - 3 - 4. Meaning: remaining eggs.",
        sub_goal="Calculate 16 - 3 - 4.",
        tool_name="Calculator_Tool",
        tool_metadata={"input_types": {"expression": "str"}},
        step_count=1,
        json_data={},
    )

    prompt = captured["prompt"]
    self.assertIn('execution = tool.execute(expression="<raw arithmetic expression>")', prompt)
    self.assertIn("Never use query=", prompt)
    self.assertIn("Return only one Python code block", prompt)
    self.assertNotIn("<Calculator>", prompt)
```

```python
def test_verifier_prompt_controls_continue_or_stop_from_memory(self):
    captured = {}

    class FakeLLM:
        def __call__(self, input_data, **kwargs):
            captured["prompt"] = input_data[0]
            return "Explanation:\nNeed one more calculation.\n\nConclusion: CONTINUE"

    from agentflow.models.memory import Memory
    from agentflow.models.verifier import Verifier

    memory = Memory()
    memory.add_action(
        1,
        "Calculator_Tool",
        "Calculate 16 - 3 - 4.",
        'execution = tool.execute(expression="16 - 3 - 4")',
        ["16 - 3 - 4 = 9"],
    )

    verifier = Verifier.__new__(Verifier)
    verifier.available_tools = ["Calculator_Tool"]
    verifier.toolbox_metadata = {"Calculator_Tool": {"input_types": {"expression": "str"}}}
    verifier.is_multimodal = False
    verifier.llm_engine_fixed = FakeLLM()

    verifier.verificate_context(
        question="Janet sells remaining eggs for $2 each. How much money?",
        image=None,
        query_analysis="Need remaining eggs then multiply by price.",
        memory=memory,
        step_count=1,
        json_data={},
    )

    prompt = captured["prompt"]
    self.assertIn("STOP only when", prompt)
    self.assertIn("CONTINUE when", prompt)
    self.assertIn("next missing numeric expression", prompt)
    self.assertIn("Conclusion: STOP", prompt)
    self.assertIn("Conclusion: CONTINUE", prompt)
```

- [x] **Step 2: Run the prompt tests to verify failures before implementation**

Run:

```bash
/home/north/vllm_test/.venv/bin/python -m unittest try_agentflow/AgentFlow/try_gsm8k_0522/tests/test_role_prompts.py -v
```

Expected before implementation: at least one assertion fails because the new contract phrases are not present yet.

---

### Task 2: Update Planner Prompts

**Files:**
- Modify: `try_agentflow/AgentFlow/agentflow/agentflow/models/planner.py:93`
- Modify: `try_agentflow/AgentFlow/agentflow/agentflow/models/planner.py:281`
- Test: `try_agentflow/AgentFlow/try_gsm8k_0522/tests/test_role_prompts.py`

- [x] **Step 1: Replace the calculator-only `analyze_query()` prompt**

In `Planner.analyze_query()`, replace the `elif calculator_only:` prompt with:

```python
query_prompt = f"""
Task: Planner analysis role for GSM8K arithmetic.

You are the first Planner call in the existing AgentFlow loop:
Planner.analyze_query -> Planner.generate_next_step -> Executor.generate_tool_command -> execute_tool_command -> Memory -> Verifier -> next Planner step.

Inputs:
- Problem: {question}
- Available tools: {self.available_tools}
- Metadata for tools: {self.toolbox_metadata}

Instructions:
1. Read the problem and identify the target unknown.
2. Extract only quantities explicitly present in the problem.
3. Note required unit conversions, for example dozen = 12 and percent = value / 100.
4. Write a compact arithmetic roadmap for later Planner steps.
5. Do not call Calculator_Tool.
6. Do not invent results for calculations that have not been executed.
7. Do not mention any tool except Calculator_Tool.

Response Format:
Concise Summary: <one sentence target unknown>
Known Quantities: <short bullet-style list of explicit quantities>
Arithmetic Roadmap: <ordered list of calculations to perform later>
Calculator Expressions To Consider: <raw expressions without results>
Additional Considerations: <unit conversion or ambiguity notes>

Rules:
- This is analysis only.
- Do not output final answer.
- Do not include `<Calculator>` tags.
- Do not write `execution = tool.execute(...)`; the Executor does that later.
"""
```

- [x] **Step 2: Replace the calculator-only `generate_next_step()` prompt**

In `Planner.generate_next_step()`, replace the `elif calculator_only:` prompt with:

```python
prompt_generate_next_step = f"""
Task: Planner next-step role for GSM8K arithmetic.

You are inside the existing AgentFlow loop:
1. Planner chooses the next single unresolved arithmetic expression.
2. Executor converts that expression into `execution = tool.execute(expression="...")`.
3. The tool result is written to Memory.
4. Verifier decides STOP or CONTINUE.
5. If CONTINUE, Planner receives updated Memory and chooses the next expression.

Context:
- Problem: {question}
- Query Analysis: {query_analysis}
- Available Tools: {self.available_tools}
- Toolbox Metadata: {self.toolbox_metadata}
- Previous Steps: {memory.get_actions()}
- Current Step: {step_count} of {max_step_count}
- Remaining Steps: {max_step_count - step_count}

Instructions:
1. Choose exactly one next single unresolved arithmetic expression needed to solve the problem.
2. Use successful Calculator_Tool results from Previous Steps when forming the next expression.
3. If no successful result exists yet, start with the first necessary expression from the problem.
4. Put the raw numeric expression in Context after `Expression to calculate:`.
5. Put a short meaning after `Meaning:`.
6. Do not include the result, an equals sign with the result, or the final answer.
7. Select only Calculator_Tool.
8. Do not use variables or words inside the expression.

Response Format:
Justification: <why this is the next required calculation>
Context: Expression to calculate: <raw numeric expression>. Meaning: <what the expression represents>.
Sub-Goal: Calculate <raw numeric expression>.
Tool Name: Calculator_Tool

Expression Rules:
- Allowed characters: digits, spaces, +, -, *, /, parentheses, decimal points.
- Convert units before writing the expression, for example 2 dozen becomes 2 * 12.
- Convert percentages before writing the expression, for example 30% becomes 30 / 100.
- Do not include units, variable names, currency symbols, commas, or natural-language phrases in the expression.

Stop Rules:
- If Memory already contains enough successful calculator results to answer the original problem, choose the final combining expression only if it has not already been executed.
- Never output STOP; Verifier is responsible for STOP or CONTINUE.

Final Rules:
- End with `Tool Name: Calculator_Tool`.
- Do not include any text after the tool name.
- Do not include `<Calculator>` tags.
- Do not write `execution = tool.execute(...)`; the Executor does that later.
"""
```

- [x] **Step 3: Run Planner prompt tests**

Run:

```bash
/home/north/vllm_test/.venv/bin/python -m unittest try_agentflow/AgentFlow/try_gsm8k_0522/tests/test_role_prompts.py -v
```

Expected after this task: Planner-specific tests pass; Executor and Verifier tests may still fail until later tasks.

---

### Task 3: Update Executor Prompt

**Files:**
- Modify: `try_agentflow/AgentFlow/agentflow/agentflow/models/executor.py:83`
- Test: `try_agentflow/AgentFlow/try_gsm8k_0522/tests/test_executor_calculator_prompt.py`
- Test: `try_agentflow/AgentFlow/try_gsm8k_0522/tests/test_role_prompts.py`

- [x] **Step 1: Replace the calculator-only `generate_tool_command()` prompt**

In `Executor.generate_tool_command()`, replace the `if tool_name == "Calculator_Tool":` prompt with:

```python
prompt_generate_tool_command = f"""
Task: Executor role for GSM8K arithmetic.

You are inside the existing AgentFlow loop after Planner.generate_next_step.
Planner has already chosen the next arithmetic sub-goal.
Your only job is to translate the Planner expression into one executable Calculator_Tool command.

Context:
- Problem: {question}
- Planner Context: {context}
- Planner Sub-Goal: {sub_goal}
- Tool Name: {tool_name}
- Tool Metadata: {tool_metadata}

Calculator_Tool contract:
- The runtime only recognizes Python commands matching `execution = tool.execute(...)`.
- Use the `expression` argument only.
- Never use `query=`.
- The expression value must be a raw arithmetic string.
- The expression must contain only digits, spaces, +, -, *, /, parentheses, and decimal points.
- Convert `×` to `*` and `÷` to `/` if they appear in Planner text.
- Remove units, words, dollar signs, percent signs, commas, and expected answers.
- Do not solve the expression yourself.
- Do not include the result.

Output Format:
Return only one Python code block and no prose:
```python
execution = tool.execute(expression="<raw arithmetic expression>")
```

Valid example:
```python
execution = tool.execute(expression="16 - 3 - 4")
```

Invalid examples:
```python
execution = tool.execute(query="16 - 3 - 4")
execution = tool.execute(expression="16 - 3 - 4 = 9")
execution = tool.execute(expression="remaining eggs: 16 - 3 - 4")
```
"""
```

- [x] **Step 2: Run Executor prompt tests**

Run:

```bash
/home/north/vllm_test/.venv/bin/python -m unittest try_agentflow/AgentFlow/try_gsm8k_0522/tests/test_executor_calculator_prompt.py try_agentflow/AgentFlow/try_gsm8k_0522/tests/test_role_prompts.py -v
```

Expected after this task: Executor tests pass and confirm the prompt requires `execution = tool.execute(expression="...")`.

---

### Task 4: Update Verifier Prompt

**Files:**
- Modify: `try_agentflow/AgentFlow/agentflow/agentflow/models/verifier.py:107`
- Test: `try_agentflow/AgentFlow/try_gsm8k_0522/tests/test_role_prompts.py`

- [x] **Step 1: Replace the calculator-only `verificate_context()` prompt**

In `Verifier.verificate_context()`, replace the `elif calculator_only:` prompt with:

```python
prompt_memory_verification = f"""
Task: Verifier role for GSM8K arithmetic.

You are inside the existing AgentFlow loop after one Calculator_Tool execution.
Your job is to decide whether Memory has enough successful calculator results to answer the original problem.

Context:
- Problem: {question}
- Available Tools: {self.available_tools}
- Toolbox Metadata: {self.toolbox_metadata}
- Initial Analysis: {query_analysis}
- Memory (Calculator calls and results): {memory.get_actions()}

Instructions:
1. Inspect Memory step by step.
2. Treat a step as successful only when:
   - tool_name is Calculator_Tool,
   - command contains `execution = tool.execute(expression="...")`,
   - result is non-empty,
   - result is not an Error string.
3. STOP only when the successful results are enough to derive the final numeric answer.
4. CONTINUE when at least one required calculation is missing.
5. If continuing, state the next missing numeric expression in the explanation.
6. Do not call tools.
7. Do not mention any tool except Calculator_Tool.
8. Do not output the final answer unless explaining that Memory is complete.

Response Format When Complete:
Explanation:
Memory contains successful calculator results for <brief chain>. The final numeric answer can be derived from <specific result or expression>.

Conclusion: STOP

Response Format When Incomplete:
Explanation:
Memory is missing the next calculation: <raw numeric expression>. This is needed to compute <meaning>.

Conclusion: CONTINUE

Final Rules:
- The response must end with exactly one of:
Conclusion: STOP
Conclusion: CONTINUE
- Do not include any text after the conclusion.
- Do not include `<Calculator>` tags.
- Do not write `execution = tool.execute(...)`; the Executor does that later.
"""
```

- [x] **Step 2: Run Verifier prompt tests**

Run:

```bash
/home/north/vllm_test/.venv/bin/python -m unittest try_agentflow/AgentFlow/try_gsm8k_0522/tests/test_role_prompts.py -v
```

Expected after this task: Verifier test passes and confirms `STOP`/`CONTINUE` behavior is defined from Memory.

---

### Task 5: Update Generator Prompts

**Files:**
- Modify: `try_agentflow/AgentFlow/agentflow/agentflow/models/planner.py:359`
- Modify: `try_agentflow/AgentFlow/agentflow/agentflow/models/planner.py:464`
- Test: `try_agentflow/AgentFlow/try_gsm8k_0522/tests/test_role_prompts.py`
- Test: `try_agentflow/AgentFlow/try_gsm8k_0522/tests/test_gsm8k_utils.py`

- [x] **Step 1: Replace calculator-only Generator prompts**

Use the same contract in both `generate_final_output()` and `generate_direct_output()` calculator-only branches:

```python
prompt_generate_final_output = f"""
Task: Generator role for GSM8K arithmetic.

You run after Verifier has stopped or after max steps.
Use only the original problem and successful Calculator_Tool results from Memory.

Context:
- Problem: {question}
- Actions Taken: {memory.get_actions()}

Instructions:
1. Read the original problem.
2. Read successful calculator results from Actions Taken.
3. Reconstruct the final answer only from executed results.
4. If Memory contains an incomplete or invalid step, ignore that step.
5. Do not invent a calculator result that is not present in Actions Taken.
6. Do not request another tool call.

Output exactly:
Planner:
1. <one short sentence summarizing the arithmetic plan>

Executor:
1. execution = tool.execute(expression="<expression>") -> <result>

Verifier:
<one sentence explaining why the final number answers the problem>

Generator:
<only the final numeric answer>

Rules:
- The Executor section is a report of completed calls, not a new command request.
- Include only successful Calculator_Tool calls from Actions Taken.
- The Generator section must contain only the final numeric answer.
- No units, prose, markdown, currency symbols, or commas in the Generator answer.
- Do not output any text after the Generator answer.
- Do not include `<Calculator>` tags.
"""
```

For `generate_direct_output()`, include `Initial Analysis: {self.query_analysis}` in the Context block in addition to the fields above.

- [x] **Step 2: Run Generator scoring utility tests**

Run:

```bash
/home/north/vllm_test/.venv/bin/python -m unittest try_agentflow/AgentFlow/try_gsm8k_0522/tests/test_gsm8k_utils.py try_agentflow/AgentFlow/try_gsm8k_0522/tests/test_role_prompts.py -v
```

Expected after this task: tests pass and extraction still supports final numeric answers.

---

### Task 6: Full Verification

**Files:**
- Test: `try_agentflow/AgentFlow/try_gsm8k_0522/tests/`
- Test: Python syntax for changed source files
- Optional smoke output: `try_agentflow/AgentFlow/try_gsm8k_0522/results/`

- [x] **Step 1: Run all GSM8K tests**

Run:

```bash
/home/north/vllm_test/.venv/bin/python -m unittest discover -s try_agentflow/AgentFlow/try_gsm8k_0522/tests -v
```

Expected: all tests pass.

- [x] **Step 2: Compile changed Python files**

Run:

```bash
/home/north/vllm_test/.venv/bin/python -m py_compile try_agentflow/AgentFlow/agentflow/agentflow/models/planner.py try_agentflow/AgentFlow/agentflow/agentflow/models/executor.py try_agentflow/AgentFlow/agentflow/agentflow/models/verifier.py
```

Expected: command exits with status 0.

- [x] **Step 3: Run smoke with vLLM service already running**

Status 2026-05-22: completed from the host environment after confirming `curl --noproxy '*' -sS --max-time 3 http://127.0.0.1:8000/v1/models` returned `Qwen3-0.6B-Instruct`.

Run:

```bash
cd /home/north/vllm_test/try_agentflow/AgentFlow/try_gsm8k_0522
bash run_smoke.sh
```

Expected:
- Results are written under `results/`.
- Summary files are written under `summary/`.
- JSON outputs show only `Calculator_Tool` in the AgentFlow workflow.
- Planner responses contain `Context`, `Sub-Goal`, and `Tool Name`.
- Executor responses contain `execution = tool.execute(expression="...")`.
- Memory contains calculator results.
- Verifier responses end with `Conclusion: STOP` or `Conclusion: CONTINUE`.

- [x] **Step 4: Inspect at least three failed examples if smoke accuracy remains low**

Run:

```bash
rg -n '"command": "No command found."|"No result was generated|Error in execute_tool_command|No matched tool' try_agentflow/AgentFlow/try_gsm8k_0522/results
```

Expected:
- If no matches appear, prompt-format failures were reduced.
- If matches appear, inspect the corresponding `action_predictor_*_response`, `tool_commander_*_response`, and `verifier_*_response` fields to decide whether the remaining issue is Planner expression quality, Executor command formatting, or Verifier stopping.

---

## Self-Review

- Spec coverage: The plan covers all requested workflow stages: Planner analysis, Planner next-step planning, Executor command generation, tool execution contract, Memory-based Verifier decision, and final Generator output.
- Placeholder scan: No implementation step contains TBD/TODO placeholders.
- Type consistency: The plan keeps the existing `Calculator_Tool`, `execution = tool.execute(expression="...")`, Memory action shape, and `Conclusion: STOP` / `Conclusion: CONTINUE` contracts.
