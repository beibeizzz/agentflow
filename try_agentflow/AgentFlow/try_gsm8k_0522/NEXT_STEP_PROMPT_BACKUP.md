# Calculator-Only Planner Next-Step Prompt Backup

This file preserves the previous calculator-only `Planner.generate_next_step` prompt before replacing angle-bracket placeholders with concrete examples.

```text
You should plan the next arithmetic expression to calculate.

Context:
- Problem: {question}
- Query Analysis: {query_analysis}
- Available Tools: {self.available_tools}
- Toolbox Metadata: {self.toolbox_metadata}
- Previous Steps: {memory.get_actions()}
Task:
Plan the next one raw arithmetic expression to calculate.

Previous Steps Notes:
- Verifier_feedback for reference to decide the next arithmetic expression.
- Do not repeat a previous expression.

Rules:
- Output only one JSON object and follow the format.
- If previous results are already to solve the problem, output the final combining expression.
- Only numbers are allowed in the <raw expression>.
- Do not add words, "=" or units inside <raw expression>.
- <raw expression> only uses +, -, *, /, parentheses, decimals.

JSON format:
{
  "calculation": "Next expression to calculate: <raw expression>"，
}
```
