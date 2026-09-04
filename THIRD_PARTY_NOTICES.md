# Third-party notices

## lupantech/AgentFlow

Task semantics, prompts, tool contracts, and experimental structure are derived
from [lupantech/AgentFlow](https://github.com/lupantech/AgentFlow) and the
reviewed experimental fork. AgentFlow is licensed under the MIT License. The
retained license text is in `LICENSES/AgentFlow-MIT.txt`.

## veRL

The distributed runtime, AgentLoop API, synchronous PPO/TransferQueue lifecycle,
FSDP/vLLM workers, GSPO implementation, checkpointing, and native logging are
provided by [verl-project/verl](https://github.com/verl-project/verl), pinned to
v0.8.0 commit `7aed6b230776f963fa09509c10d9c3a767d1102c`.
veRL is licensed under the Apache License 2.0; the root `LICENSE` contains those
terms. The local TaskRunner and Trainer adapter follow the public v0.8.0 API.

## Data

GSM8K is MIT-licensed. Ticket data is deterministic synthetic data with no real
customer or business records. Source, transformation, row-count, and hash
details are maintained in `data/README.md`.
