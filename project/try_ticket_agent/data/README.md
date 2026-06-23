# Ticket data contract

`blueprints/` is deterministic generator output. `generated/` is intentionally absent until API synthesis is run. Do not train on blueprint canonical requests as a silent substitute.

Each accepted JSONL row contains public `user_request`, isolated `episode_id`, curriculum/lookup mode, 6-10 ticket `initial_state`, hidden one-field `goal_spec`, and `max_steps`. It never contains reference actions. Direct/indirect ratio is 80/20; indirect lookup alternates customer and order identifiers and never exposes the target ticket ID.

Recreate and validate:

```bash
python try_ticket_agent/scripts/generate_blueprints.py --seed 42 --smoke 32 --train 2500 --validation 256 --test 512 --output-dir try_ticket_agent/data/blueprints
cp try_ticket_agent/config_synthesis.example.yaml try_ticket_agent/config_synthesis.yaml
DEEPSEEK_API_KEY=... python try_ticket_agent/scripts/synthesize_dataset.py --config try_ticket_agent/config_synthesis.yaml
python try_ticket_agent/scripts/validate_dataset.py --blueprints try_ticket_agent/data/blueprints
python try_ticket_agent/scripts/validate_dataset.py --dataset try_ticket_agent/data/generated
```

The synthesis judge is an LLM judge used only during data synthesis; training and evaluation consume frozen JSONL data and never synthesize online. Local synthesis requires `openai`, `PyYAML`, `DEEPSEEK_API_KEY`, network access to the configured base URL, and a copied `config_synthesis.yaml`; it does not require a local GPU and may incur API charges.

The manifest records counts and SHA-256 hashes. Validation checks split isolation, duplicate IDs/states/request-goal signatures, introduced identifiers/enums, indirect target leakage, multiple mutations, tool hints, completion intent, and real tool/verifier reference execution.
