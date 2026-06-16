# Calculator Prompt Rewriter

This tool rewrites the selected GSM8K training questions into a fixed format for the calculator-only AgentFlow Planner:

```text
Known facts:
- ...

Question:
- ...
```

The generated question contains no solution instructions, equations, intermediate values, or final answer. The original solution and gold answer are sent only as hidden reference context for generation and independent review.

## Models

- `deepseek-v4-flash`, thinking disabled: creates a candidate rewrite.
- `deepseek-v4-pro`, thinking enabled with high reasoning effort: judges semantic equivalence and calculator suitability.

The client uses the OpenAI-compatible endpoint `https://api.deepseek.com` and JSON Output mode.

## API Key

PowerShell:

```powershell
$env:DEEPSEEK_API_KEY="your-real-key"
```

Bash:

```bash
export DEEPSEEK_API_KEY="your-real-key"
```

The key is read only from the environment. It is not written to configuration, logs, progress records, or output datasets.

## Smoke Run

Run a small paid sample before processing the full dataset:

```powershell
C:\all_software\anaconda3\envs\all-in-rag\python.exe -B `
  project\try_gsm8k_0522\rewrite_calculator_prompts\rewrite_dataset.py `
  --limit 5 `
  --concurrency 1
```

Full run:

```powershell
C:\all_software\anaconda3\envs\all-in-rag\python.exe -B `
  project\try_gsm8k_0522\rewrite_calculator_prompts\rewrite_dataset.py
```

Use `--no-resume` to discard existing progress and start the selected range again. The default is to resume.

## Validation

Each candidate must pass:

1. Strict two-section formatting.
2. Local checks for equations, solution markers, instructions, new numbers, missing numbers, and leaked solution-only values.
3. Independent `deepseek-v4-pro` review for semantic equivalence, unchanged relation direction and target, calculator-only solvability, and a reasonable solution using no more than three meaningful calls.

Failed checks are returned to the rewrite model. A sample is rejected after three unsuccessful rewrite attempts.

## Outputs

The default output directory is `rewrite_calculator_prompts/outputs/`:

- `gsm8k_train_calculator_structured.json`: accepted source records with updated `question` and synchronized `query`.
- `progress.jsonl`: final status for each processed source index; used for resume.
- `rejected.jsonl`: rejected source records and concise failure reasons.
- `summary.json`: accepted/rejected counts and reported token usage.

No model reasoning content is stored.
