# Ticket Agent Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the approved Ticket Agent hardening gaps before remote experiments while preserving the existing turn-level GSPO objective and task contract.

**Architecture:** Keep changes task-scoped except for the shared AgentFlow rollout containment fix in `flowgrpo_light/agentflow_rollout.py`. Add small pure helpers for fingerprinting, output-dir resolution, reward-mode validation, metrics summaries, and GPU snapshots so tests can verify behavior without launching models, GPUs, network calls, or vLLM.

**Tech Stack:** Python 3.11, `unittest`, YAML configs, existing Ticket AgentFlow task, existing `flowgrpo_light` GSPO utilities, Windows/PowerShell local execution with `conda run -n all-in-rag`.

---

## File Structure

- Modify: `project/try_ticket_agent/data_synthesis/blueprints.py`
  - Add `blueprint_fingerprint(blueprint: EpisodeBlueprint) -> str`, a stable SHA-256 digest over the complete canonical `blueprint.to_dict()` payload.
- Modify: `project/try_ticket_agent/data_synthesis/pipeline.py`
  - Add progress schema versioning and strict resume validation against the current blueprint set.
  - Keep final dataset ordered by the current input order and prevent stale records from entering output.
- Modify: `project/try_ticket_agent/data_synthesis/validators.py`
  - Narrow enum extraction to assignment-shaped phrases only.
- Modify: `project/try_gsm8k_0522/flowgrpo_light/agentflow_rollout.py`
  - Move question extraction, row-aware reset, planner-sample clearing, solve, and result adaptation into the worker exception boundary.
- Modify: `project/try_ticket_agent/flowgrpo_general_2x40g/eval_ticket_agent.py`
  - Add an explicit `--output-dir` CLI override and a `resolve_output_dir()` helper.
- Modify: `project/try_ticket_agent/flowgrpo_general_2x40g/run_eval_learnable_general_2x40g.sh`
  - Pass optional `OUTPUT_DIR` only when explicitly set.
- Modify: `project/try_ticket_agent/flowgrpo_general_2x40g/train_ticket_gspo.py`
  - Enforce binary reward mode before model construction.
  - Add pure metric helpers and use them in each metrics record and final summary.
- Modify: `project/try_ticket_agent/README.md`
  - Replace misleading “offline-only synthesis judge” wording with “LLM judge used only during data synthesis.”
- Modify: `project/try_ticket_agent/data/README.md`
  - Mirror the same judge wording and local synthesis prerequisites if currently ambiguous.
- Modify tests:
  - `project/try_ticket_agent/tests/test_synthesis_pipeline.py`
  - `project/try_ticket_agent/tests/test_ticket_rollout.py`
  - `project/try_ticket_agent/tests/test_ticket_eval.py`
  - `project/try_ticket_agent/tests/test_ticket_gspo.py`
  - `project/try_ticket_agent/tests/test_remote_configs.py`

---

## Task 1: Strict synthesis resume identity

**Files:**
- Modify: `project/try_ticket_agent/data_synthesis/blueprints.py`
- Modify: `project/try_ticket_agent/data_synthesis/pipeline.py`
- Test: `project/try_ticket_agent/tests/test_synthesis_pipeline.py`

- [ ] **Step 1: Write failing tests for progress fingerprinting and strict resume**

Append these tests to `SynthesisPipelineTests` in `project/try_ticket_agent/tests/test_synthesis_pipeline.py`:

```python
    def test_resume_rejects_same_episode_id_with_changed_blueprint_fingerprint(self) -> None:
        original = generate_blueprint(seed=42, split="train", index=0)
        changed = generate_blueprint(seed=99, split="train", index=0)
        client = FakeClient(
            [{"user_request": original.canonical_request}],
            [{"accepted": True, "reasons": []}],
        )
        paths = PipelinePaths.in_directory(self.output_dir)
        TicketSynthesisPipeline(client).run([original], paths=paths, resume=True)

        second_client = FakeClient(
            [{"user_request": changed.canonical_request}],
            [{"accepted": True, "reasons": []}],
        )
        with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
            TicketSynthesisPipeline(second_client).run([changed], paths=paths, resume=True)
        self.assertEqual(second_client.rewrite_calls, [])
        self.assertEqual(second_client.judge_calls, [])

    def test_resume_rejects_extra_stale_progress_record(self) -> None:
        first = generate_blueprint(seed=42, split="train", index=0)
        second = generate_blueprint(seed=42, split="train", index=1)
        client = FakeClient(
            [
                {"user_request": first.canonical_request},
                {"user_request": second.canonical_request},
            ],
            [
                {"accepted": True, "reasons": []},
                {"accepted": True, "reasons": []},
            ],
        )
        paths = PipelinePaths.in_directory(self.output_dir)
        TicketSynthesisPipeline(client).run([first, second], paths=paths, resume=True)

        resume_client = FakeClient([], [])
        with self.assertRaisesRegex(RuntimeError, "stale progress"):
            TicketSynthesisPipeline(resume_client).run([first], paths=paths, resume=True)
        self.assertEqual(resume_client.rewrite_calls, [])
        self.assertEqual(resume_client.judge_calls, [])

    def test_resume_false_clears_stale_progress_and_recomputes(self) -> None:
        original = generate_blueprint(seed=42, split="train", index=0)
        changed = generate_blueprint(seed=99, split="train", index=0)
        paths = PipelinePaths.in_directory(self.output_dir)
        TicketSynthesisPipeline(
            FakeClient(
                [{"user_request": original.canonical_request}],
                [{"accepted": True, "reasons": []}],
            )
        ).run([original], paths=paths, resume=True)

        client = FakeClient(
            [{"user_request": changed.canonical_request}],
            [{"accepted": True, "reasons": []}],
        )
        summary = TicketSynthesisPipeline(client).run([changed], paths=paths, resume=False)
        rows = [json.loads(line) for line in paths.dataset.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["api_calls"], 2)
        self.assertEqual([row["episode_id"] for row in rows], [changed.episode_id])
```

- [ ] **Step 2: Run focused tests and confirm they fail**

Run:

```powershell
conda run -n all-in-rag python -m unittest project.try_ticket_agent.tests.test_synthesis_pipeline -q
```

Expected: fail because `blueprint_fingerprint` and strict progress validation do not exist yet, or because stale progress is silently accepted.

- [ ] **Step 3: Add blueprint fingerprint helper**

In `project/try_ticket_agent/data_synthesis/blueprints.py`, add this function near `_digest()`:

```python
def blueprint_fingerprint(blueprint: EpisodeBlueprint) -> str:
    return _digest(blueprint.to_dict())
```

- [ ] **Step 4: Add progress schema version and strict loader**

In `project/try_ticket_agent/data_synthesis/pipeline.py`, update the import and add constants/helpers:

```python
from .blueprints import blueprint_fingerprint, execute_reference_actions


SYNTHESIS_PROGRESS_SCHEMA_VERSION = 2
```

Replace the existing resume load in `TicketSynthesisPipeline.run()`:

```python
        completed = _load_progress(paths.progress, blueprints) if resume else {}
```

In the `save()` nested function, add schema and fingerprint fields:

```python
                "schema_version": SYNTHESIS_PROGRESS_SCHEMA_VERSION,
                "blueprint_fingerprint": blueprint_fingerprint(blueprint),
```

Replace `_load_progress()` with:

```python
def _load_progress(path: Path, blueprints: list[EpisodeBlueprint]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    expected = {blueprint.episode_id: blueprint_fingerprint(blueprint) for blueprint in blueprints}
    completed: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            episode_id = str(payload["episode_id"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Invalid progress record at {path}:{line_number}") from exc
        if episode_id not in expected:
            raise RuntimeError(
                f"stale progress record at {path}:{line_number} for episode_id={episode_id}; "
                "use resume: false or a new output directory"
            )
        schema_version = int(payload.get("schema_version", 1))
        if schema_version != SYNTHESIS_PROGRESS_SCHEMA_VERSION:
            raise RuntimeError(
                f"progress schema mismatch at {path}:{line_number} for episode_id={episode_id}; "
                "use resume: false or a new output directory"
            )
        stored_fingerprint = str(payload.get("blueprint_fingerprint", ""))
        if stored_fingerprint != expected[episode_id]:
            raise RuntimeError(
                f"fingerprint mismatch at {path}:{line_number} for episode_id={episode_id}; "
                "use resume: false or a new output directory"
            )
        completed[episode_id] = payload
    return completed
```

- [ ] **Step 5: Preserve current-input ordering for output**

In `TicketSynthesisPipeline.run()`, replace:

```python
        ordered = sorted(completed.values(), key=lambda item: int(item["source_index"]))
```

with:

```python
        ordered = [completed[blueprint.episode_id] for blueprint in blueprints if blueprint.episode_id in completed]
```

- [ ] **Step 6: Run focused synthesis tests**

Run:

```powershell
conda run -n all-in-rag python -m unittest project.try_ticket_agent.tests.test_synthesis_pipeline -q
```

Expected: all synthesis pipeline tests pass.

- [ ] **Step 7: Commit Task 1**

Run:

```powershell
git add project/try_ticket_agent/data_synthesis/blueprints.py project/try_ticket_agent/data_synthesis/pipeline.py project/try_ticket_agent/tests/test_synthesis_pipeline.py
git commit -m "fix(ticket-agent): validate synthesis resume identity"
```

---

## Task 2: Shared AgentFlow rollout exception containment

**Files:**
- Modify: `project/try_gsm8k_0522/flowgrpo_light/agentflow_rollout.py`
- Test: `project/try_ticket_agent/tests/test_ticket_rollout.py`

- [ ] **Step 1: Write failing rollout containment tests**

Append these tests to `TicketRolloutHookTests` in `project/try_ticket_agent/tests/test_ticket_rollout.py`:

```python
    def test_question_getter_failure_returns_invalid_rollout_without_aborting_group(self) -> None:
        class FakeSolver:
            def __init__(self):
                self.planner = types.SimpleNamespace(llm_engine=None)

            def solve(self, question):
                raise AssertionError("solve must not run when question getter fails")

        runner = AgentFlowBatchRolloutRunner(
            policy=ExactIdPolicy(),
            solver_factory=FakeSolver,
            reset_solver=lambda solver, row: None,
            question_getter=lambda row: (_ for _ in ()).throw(KeyError("user_request")),
            rollout_concurrency=2,
            planner_batch_size=2,
        )
        try:
            groups = runner.run_batch([{"episode_id": "ticket-tr-000001"}], group_size=2)
        finally:
            runner.close()

        self.assertEqual(len(groups[0]), 2)
        self.assertTrue(all(item.reward == 0.0 for item in groups[0]))
        self.assertTrue(all(item.valid_for_training is False for item in groups[0]))
        self.assertTrue(all("KeyError" in item.errors[0] for item in groups[0]))

    def test_reset_failure_returns_invalid_rollout_without_aborting_other_rows(self) -> None:
        class FakeSolver:
            def __init__(self):
                self.planner = types.SimpleNamespace(llm_engine=None)

            def solve(self, question):
                self.planner.llm_engine("ticket planner prompt", system_prompt="planner system")
                return {"question": question}

        def reset_solver(solver, row):
            if row["episode_id"] == "bad":
                raise RuntimeError("reset failed")

        def adapt(solver, row, result, samples):
            return RolloutResult(
                reward=1.0,
                answer=row["episode_id"],
                samples=list(samples),
                memory={},
                valid_for_training=True,
            )

        runner = AgentFlowBatchRolloutRunner(
            policy=ExactIdPolicy(),
            solver_factory=FakeSolver,
            reset_solver=reset_solver,
            result_adapter=adapt,
            question_getter=lambda row: row["user_request"],
            rollout_concurrency=2,
            planner_batch_size=2,
        )
        rows = [
            {"episode_id": "bad", "user_request": "bad request"},
            {"episode_id": "good", "user_request": "good request"},
        ]
        try:
            groups = runner.run_batch(rows, group_size=1)
        finally:
            runner.close()

        self.assertEqual(groups[0][0].valid_for_training, False)
        self.assertIn("RuntimeError: reset failed", groups[0][0].errors)
        self.assertEqual(groups[1][0].valid_for_training, True)
        self.assertEqual(groups[1][0].reward, 1.0)
        self.assertEqual(groups[1][0].samples[0].response_token_ids, [201, 202])
```

- [ ] **Step 2: Run focused rollout tests and confirm they fail**

Run:

```powershell
conda run -n all-in-rag python -m unittest project.try_ticket_agent.tests.test_ticket_rollout -q
```

Expected: fail because `question_getter` or reset exceptions escape through `Future.result()`.

- [ ] **Step 3: Move all worker pre-solve operations into the exception boundary**

In `project/try_gsm8k_0522/flowgrpo_light/agentflow_rollout.py`, replace `AgentFlowRolloutWorker.run()` with:

```python
    def run(self, row: dict[str, Any]) -> RolloutResult:
        from flowgrpo.reward import compute_result_reward

        try:
            question = str(self.question_getter(row))
            gold_answer = str(row.get("result") or row.get("gold_answer") or row.get("extra_info", {}).get("gold_answer"))
            _call_reset_solver(self.reset_solver, self.solver, row)
            self.policy_engine.clear()
            result = self.solver.solve(question)
            if self.result_adapter is not None:
                return self.result_adapter(
                    self.solver,
                    row,
                    result,
                    list(self.policy_engine.samples),
                )
            reward, predicted_answer = compute_result_reward(result, gold_answer)
            answer: Any = result.get("direct_output") or result.get("final_output") or result.get("base_response")
            return RolloutResult(
                reward=reward,
                answer=str(answer if answer is not None else predicted_answer),
                samples=list(self.policy_engine.samples),
                memory=result.get("memory") or {},
                query_analysis=str(result.get("query_analysis") or ""),
                errors=[],
            )
        except Exception as exc:
            return RolloutResult(
                reward=0.0,
                answer="",
                samples=list(self.policy_engine.samples),
                memory={},
                query_analysis="",
                errors=[f"{type(exc).__name__}: {exc}"],
                valid_for_training=False,
            )
```

- [ ] **Step 4: Run rollout tests and shared GSM AgentFlow tests**

Run:

```powershell
conda run -n all-in-rag python -m unittest project.try_ticket_agent.tests.test_ticket_rollout project.try_gsm8k_0522.tests.test_flowgrpo_light_agentflow -q
```

Expected: all listed tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add project/try_gsm8k_0522/flowgrpo_light/agentflow_rollout.py project/try_ticket_agent/tests/test_ticket_rollout.py
git commit -m "fix(flowgrpo): contain agentflow rollout hook failures"
```

---

## Task 3: Evaluation output isolation

**Files:**
- Modify: `project/try_ticket_agent/flowgrpo_general_2x40g/eval_ticket_agent.py`
- Modify: `project/try_ticket_agent/flowgrpo_general_2x40g/run_eval_learnable_general_2x40g.sh`
- Test: `project/try_ticket_agent/tests/test_ticket_eval.py`
- Test: `project/try_ticket_agent/tests/test_remote_configs.py`

- [ ] **Step 1: Write failing tests for mode-specific output dirs and explicit override**

In `project/try_ticket_agent/tests/test_ticket_eval.py`, update the import:

```python
from pathlib import Path

from try_ticket_agent.flowgrpo_general_2x40g.eval_ticket_agent import resolve_output_dir, summarize_results
```

Append:

```python
    def test_resolve_output_dir_separates_baseline_and_adapter_defaults(self) -> None:
        config = {
            "output_dir": "try_ticket_agent/flowgrpo_general_2x40g/outputs/eval_adapter",
        }
        self.assertEqual(
            resolve_output_dir(config, mode="baseline", explicit_output_dir=None),
            Path("try_ticket_agent/flowgrpo_general_2x40g/outputs/eval_baseline"),
        )
        self.assertEqual(
            resolve_output_dir(config, mode="adapter", explicit_output_dir=None),
            Path("try_ticket_agent/flowgrpo_general_2x40g/outputs/eval_adapter"),
        )

    def test_resolve_output_dir_honors_explicit_override(self) -> None:
        config = {"output_dir": "configured/adapter"}
        self.assertEqual(
            resolve_output_dir(config, mode="baseline", explicit_output_dir=Path("manual/out")),
            Path("manual/out"),
        )
```

In `project/try_ticket_agent/tests/test_remote_configs.py`, append to `RemoteConfigTests`:

```python
    def test_eval_script_supports_explicit_output_dir_without_forcing_shared_default(self) -> None:
        eval_script = ROOT / "flowgrpo_general_2x40g" / "run_eval_learnable_general_2x40g.sh"
        text = eval_script.read_text(encoding="utf-8")
        self.assertIn("OUTPUT_DIR", text)
        self.assertIn("--output-dir", text)
```

- [ ] **Step 2: Run focused eval tests and confirm they fail**

Run:

```powershell
conda run -n all-in-rag python -m unittest project.try_ticket_agent.tests.test_ticket_eval project.try_ticket_agent.tests.test_remote_configs -q
```

Expected: fail because `resolve_output_dir()` and `--output-dir` do not exist.

- [ ] **Step 3: Add `resolve_output_dir()` and CLI argument**

In `project/try_ticket_agent/flowgrpo_general_2x40g/eval_ticket_agent.py`, add:

```python
def resolve_output_dir(
    config: dict[str, Any],
    *,
    mode: str,
    explicit_output_dir: Path | None,
) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir
    base = Path("try_ticket_agent/flowgrpo_general_2x40g/outputs")
    if mode == "baseline":
        return base / "eval_baseline"
    return Path(config_value(config, "output_dir", str(base / "eval_adapter")))
```

In `parse_args()`, add:

```python
    parser.add_argument("--output-dir", type=Path, default=None)
```

In `main()`, replace the current `output_dir = ...` line with:

```python
    output_dir = resolve_output_dir(config, mode=mode, explicit_output_dir=args.output_dir)
```

- [ ] **Step 4: Pass optional shell override**

In `project/try_ticket_agent/flowgrpo_general_2x40g/run_eval_learnable_general_2x40g.sh`, replace the final Python invocation with an argument array:

```bash
ARGS=(--config "$CONFIG_FILE" --eval-mode "$EVAL_MODE" --adapter-path "$ADAPTER_PATH")
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi
python try_ticket_agent/flowgrpo_general_2x40g/eval_ticket_agent.py "${ARGS[@]}"
```

- [ ] **Step 5: Run focused eval/config tests**

Run:

```powershell
conda run -n all-in-rag python -m unittest project.try_ticket_agent.tests.test_ticket_eval project.try_ticket_agent.tests.test_remote_configs -q
```

Expected: all listed tests pass.

- [ ] **Step 6: Commit Task 3**

Run:

```powershell
git add project/try_ticket_agent/flowgrpo_general_2x40g/eval_ticket_agent.py project/try_ticket_agent/flowgrpo_general_2x40g/run_eval_learnable_general_2x40g.sh project/try_ticket_agent/tests/test_ticket_eval.py project/try_ticket_agent/tests/test_remote_configs.py
git commit -m "fix(ticket-agent): isolate eval outputs by mode"
```

---

## Task 4: Training reward-mode enforcement and diagnostic metrics

**Files:**
- Modify: `project/try_ticket_agent/flowgrpo_general_2x40g/train_ticket_gspo.py`
- Test: `project/try_ticket_agent/tests/test_ticket_gspo.py`

- [ ] **Step 1: Write failing tests for reward-mode validation, metrics shape, and GPU snapshot**

Update the import from `train_ticket_gspo` in `project/try_ticket_agent/tests/test_ticket_gspo.py`:

```python
    build_training_record,
    gpu_memory_snapshot,
    resolve_reward_mode,
    summarize_rewards,
```

Append:

```python
    def test_reward_mode_must_be_binary(self) -> None:
        self.assertEqual(resolve_reward_mode({"reward_mode": "binary"}), "binary")
        with self.assertRaisesRegex(SystemExit, "reward_mode must be binary"):
            resolve_reward_mode({"reward_mode": "dense"})

    def test_summarize_rewards_uses_population_statistics(self) -> None:
        summary = summarize_rewards([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean"], 0.5)
        self.assertEqual(summary["min"], 0.0)
        self.assertEqual(summary["max"], 1.0)
        self.assertEqual(summary["std"], 0.5)

    def test_gpu_memory_snapshot_handles_cpu_only_torch(self) -> None:
        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        fake_torch = types.SimpleNamespace(cuda=FakeCuda())
        self.assertEqual(gpu_memory_snapshot(fake_torch), {"cuda": False})

    def test_training_record_exposes_remote_diagnostics_at_top_level(self) -> None:
        groups = [[rollout(1.0, 2), rollout(0.0, 1), rollout(0.0, 1, valid=False)]]
        flat, advantages, reward_groups, advantage_groups = flatten_rollout_groups(groups)
        record = build_training_record(
            step=1,
            epoch=0,
            row_index=0,
            batch=[{"episode_id": "ticket-tr-000001"}],
            groups=groups,
            rollouts=flat,
            advantages=advantages,
            reward_groups=reward_groups,
            advantage_groups=advantage_groups,
            stats={
                "loss": 0.25,
                "ratio_mean": 1.001,
                "ratio_min": 0.999,
                "ratio_max": 1.004,
                "clip_fraction": 0.125,
                "approx_kl": 0.002,
            },
            clip_low=0.001,
            clip_high=0.003,
            policy_epochs=2,
            step_elapsed_s=1.25,
            rollout_elapsed_s=0.75,
            train_elapsed_s=0.5,
            gpu_memory={"cuda": False},
        )
        self.assertEqual(record["status"], "update")
        self.assertEqual(record["reward_count"], 3)
        self.assertEqual(record["reward_mean"], 1 / 3)
        self.assertIn("reward_std", record)
        self.assertEqual(record["valid_rollout_count"], 2)
        self.assertEqual(record["infrastructure_failure_count"], 1)
        self.assertEqual(record["response_token_count"], 6)
        self.assertEqual(record["policy_epochs"], 2)
        self.assertEqual(record["loss"], 0.25)
        self.assertEqual(record["ratio_mean"], 1.001)
        self.assertEqual(record["clip_fraction"], 0.125)
        self.assertEqual(record["approx_kl"], 0.002)
        self.assertEqual(record["gpu_memory"], {"cuda": False})
        self.assertIn("train_stats", record)
```

- [ ] **Step 2: Run focused GSPO tests and confirm they fail**

Run:

```powershell
conda run -n all-in-rag python -m unittest project.try_ticket_agent.tests.test_ticket_gspo -q
```

Expected: fail because the helper functions do not exist.

- [ ] **Step 3: Add pure helpers to `train_ticket_gspo.py`**

Add imports:

```python
import math
```

Add these helpers above `main()`:

```python
def resolve_reward_mode(config: dict[str, Any]) -> str:
    mode = str(config_value(config, "reward_mode", "binary"))
    if mode != "binary":
        raise SystemExit(f"reward_mode must be binary, got: {mode}")
    return mode


def summarize_rewards(reward_groups: list[list[float]]) -> dict[str, float | int]:
    values = [float(value) for group in reward_groups for value in group]
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def gpu_memory_snapshot(torch_module: Any) -> dict[str, Any]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return {"cuda": False}
    device = cuda.current_device()
    return {
        "cuda": True,
        "device": int(device),
        "allocated": int(cuda.memory_allocated(device)),
        "reserved": int(cuda.memory_reserved(device)),
        "max_allocated": int(cuda.max_memory_allocated(device)),
        "max_reserved": int(cuda.max_memory_reserved(device)),
    }


def build_training_record(
    *,
    step: int,
    epoch: int,
    row_index: int,
    batch: list[dict[str, Any]],
    groups: list[list[RolloutResult]],
    rollouts: list[RolloutResult],
    advantages: list[float],
    reward_groups: list[list[float]],
    advantage_groups: list[list[float | None]],
    stats: dict[str, Any],
    clip_low: float,
    clip_high: float,
    policy_epochs: int,
    step_elapsed_s: float,
    rollout_elapsed_s: float,
    train_elapsed_s: float,
    gpu_memory: dict[str, Any],
) -> dict[str, Any]:
    flat_group = [item for group in groups for item in group]
    reward_summary = summarize_rewards(reward_groups)
    record = {
        "step": step,
        "epoch": epoch,
        "row_index": row_index,
        "episode_ids": [row["episode_id"] for row in batch],
        "reward_groups": reward_groups,
        "advantage_groups": advantage_groups,
        "reward_count": reward_summary["count"],
        "reward_mean": reward_summary["mean"],
        "reward_std": reward_summary["std"],
        "reward_min": reward_summary["min"],
        "reward_max": reward_summary["max"],
        "valid_rollout_count": len(rollouts),
        "infrastructure_failure_count": len(flat_group) - len(rollouts),
        "invalid_rollout_count": len(flat_group) - len(rollouts),
        "nonzero_trajectory_count": sum(abs(value) >= 1e-8 for value in advantages),
        "nonzero_turn_count": sum(
            len(rollout.samples)
            for rollout, advantage in zip(rollouts, advantages, strict=True)
            if abs(advantage) >= 1e-8
        ),
        "response_token_count": sum(
            len(sample.response_token_ids or []) for rollout in rollouts for sample in rollout.samples
        ),
        "token_count": sum(
            len(sample.response_token_ids or []) for rollout in rollouts for sample in rollout.samples
        ),
        "zero_variance_group_count": sum(
            all(value is None or abs(value) < 1e-8 for value in group)
            for group in advantage_groups
        ),
        "status": "update" if stats else "skip_no_advantage",
        "loss": stats.get("loss") if stats else None,
        "ratio_mean": stats.get("ratio_mean") if stats else None,
        "ratio_min": stats.get("ratio_min") if stats else None,
        "ratio_max": stats.get("ratio_max") if stats else None,
        "clip_fraction": stats.get("clip_fraction") if stats else None,
        "approx_kl": stats.get("approx_kl") if stats else None,
        "policy_epochs": policy_epochs,
        "train_stats": stats,
        "clip_range_low": clip_low,
        "clip_range_high": clip_high,
        "rollout_elapsed_s": round(rollout_elapsed_s, 3),
        "train_elapsed_s": round(train_elapsed_s, 3),
        "elapsed_s": round(step_elapsed_s, 3),
        "gpu_memory": gpu_memory,
    }
    return record
```

- [ ] **Step 4: Wire helpers into `main()` before model construction**

After loading config in `main()`, add:

```python
    resolve_reward_mode(config)
```

Around rollout and train timing, replace the current step loop body timing:

```python
                step_started = time.time()
                rollout_started = time.time()
                groups = runner.run_batch(batch, group_size=group_size)
                rollout_elapsed_s = time.time() - rollout_started
                rollouts, advantages, reward_groups, advantage_groups = flatten_rollout_groups(groups)
                train_started = time.time()
                policy_epochs = int(args.policy_epochs if args.policy_epochs is not None else config_value(config, "policy_epochs", 2))
                stats = train_step_grpo(
                    policy=policy,
                    optimizer=optimizer,
                    rollouts=rollouts,
                    advantages=advantages,
                    clip_range_low=clip_low,
                    clip_range_high=clip_high,
                    max_grad_norm=float(config_value(config, "max_grad_norm", 1.0)),
                    logprob_micro_batch_size=int(config_value(config, "logprob_micro_batch_size", 8)),
                    policy_epochs=policy_epochs,
                )
                train_elapsed_s = time.time() - train_started
                step += 1
                record = build_training_record(
                    step=step,
                    epoch=epoch,
                    row_index=row_index,
                    batch=batch,
                    groups=groups,
                    rollouts=rollouts,
                    advantages=advantages,
                    reward_groups=reward_groups,
                    advantage_groups=advantage_groups,
                    stats=stats,
                    clip_low=clip_low,
                    clip_high=clip_high,
                    policy_epochs=policy_epochs,
                    step_elapsed_s=time.time() - step_started,
                    rollout_elapsed_s=rollout_elapsed_s,
                    train_elapsed_s=train_elapsed_s,
                    gpu_memory=gpu_memory_snapshot(torch),
                )
```

- [ ] **Step 5: Expand final summary**

Replace the final `train_summary.json` payload with:

```python
        json.dumps(
            {
                "steps": step,
                "rows": len(rows),
                "elapsed_s": round(time.time() - started_at, 3),
                "final_adapter": str(final_adapter),
                "clip_range_low": clip_low,
                "clip_range_high": clip_high,
                "policy_epochs": int(args.policy_epochs if args.policy_epochs is not None else config_value(config, "policy_epochs", 2)),
            },
            ensure_ascii=False,
            indent=2,
        ),
```

- [ ] **Step 6: Export helper names**

Add to `__all__`:

```python
    "build_training_record",
    "gpu_memory_snapshot",
    "resolve_reward_mode",
    "summarize_rewards",
```

- [ ] **Step 7: Run focused GSPO tests**

Run:

```powershell
conda run -n all-in-rag python -m unittest project.try_ticket_agent.tests.test_ticket_gspo -q
```

Expected: all Ticket GSPO tests pass.

- [ ] **Step 8: Commit Task 4**

Run:

```powershell
git add project/try_ticket_agent/flowgrpo_general_2x40g/train_ticket_gspo.py project/try_ticket_agent/tests/test_ticket_gspo.py
git commit -m "fix(ticket-agent): enforce binary mode and log train diagnostics"
```

---

## Task 5: Natural-language validator precision

**Files:**
- Modify: `project/try_ticket_agent/data_synthesis/validators.py`
- Test: `project/try_ticket_agent/tests/test_synthesis_pipeline.py`

- [ ] **Step 1: Write failing validator precision tests**

Append to `SynthesisPipelineTests`:

```python
    def test_validator_allows_field_reference_without_assignment_enum_capture(self) -> None:
        blueprint = generate_blueprint(seed=42, split="train", index=2)
        target_id = blueprint.goal_spec["target_ticket_id"]
        value = blueprint.goal_spec["value"]
        request = (
            f"For ticket {target_id}, update the status of the ticket so it becomes {value}, "
            "then finish the request."
        )
        result = validate_candidate(blueprint, request)
        self.assertNotIn("introduced_enum", result.codes)
        self.assertTrue(result.ok, result.messages)

    def test_validator_still_rejects_explicit_unsupported_assignment_enum(self) -> None:
        blueprint = generate_blueprint(seed=42, split="train", index=2)
        target_id = blueprint.goal_spec["target_ticket_id"]
        request = f"For ticket {target_id}, set status to pending, then finish the request."
        result = validate_candidate(blueprint, request)
        self.assertIn("introduced_enum", result.codes)
```

- [ ] **Step 2: Run validator-focused tests and confirm they fail**

Run:

```powershell
conda run -n all-in-rag python -m unittest project.try_ticket_agent.tests.test_synthesis_pipeline -q
```

Expected: fail because the current regex captures `of` after `status`.

- [ ] **Step 3: Narrow enum regexes to assignment-shaped phrases**

In `project/try_ticket_agent/data_synthesis/validators.py`, replace `patterns = ...` with:

```python
    patterns = {
        "priority": r"(?:set|change|make|mark|update)?\s*priority\s+(?:to|as|is|becomes?)\s+([a-z_]+)",
        "assigned_team": r"(?:set|change|make|mark|update)?\s*(?:assigned_team|assigned team|team)\s+(?:to|as|is|becomes?)\s+([a-z_]+)",
        "status": r"(?:set|change|make|mark|update)?\s*status\s+(?:to|as|is|becomes?)\s+([a-z_]+)",
    }
```

- [ ] **Step 4: Run synthesis tests**

Run:

```powershell
conda run -n all-in-rag python -m unittest project.try_ticket_agent.tests.test_synthesis_pipeline -q
```

Expected: all synthesis tests pass.

- [ ] **Step 5: Commit Task 5**

Run:

```powershell
git add project/try_ticket_agent/data_synthesis/validators.py project/try_ticket_agent/tests/test_synthesis_pipeline.py
git commit -m "fix(ticket-agent): narrow synthesis enum validation"
```

---

## Task 6: Documentation wording and local synthesis clarity

**Files:**
- Modify: `project/try_ticket_agent/README.md`
- Modify: `project/try_ticket_agent/data/README.md`
- Test: `project/try_ticket_agent/tests/test_remote_configs.py`

- [ ] **Step 1: Update documentation tests**

In `project/try_ticket_agent/tests/test_remote_configs.py`, replace the old required statement:

```python
            "offline-only synthesis judge",
```

with:

```python
            "LLM judge used only during data synthesis",
```

Append this assertion to `test_readmes_document_reproducible_no_sft_workflow()`:

```python
        self.assertNotIn("offline-only synthesis judge", text)
        self.assertIn("DEEPSEEK_API_KEY", text)
        self.assertIn("config_synthesis.yaml", text)
        self.assertIn("scripts/synthesize_dataset.py --config", text)
```

- [ ] **Step 2: Run remote config tests and confirm they fail**

Run:

```powershell
conda run -n all-in-rag python -m unittest project.try_ticket_agent.tests.test_remote_configs -q
```

Expected: fail because README still contains the old wording or lacks local synthesis prerequisite wording.

- [ ] **Step 3: Update README wording**

In `project/try_ticket_agent/README.md`, replace:

```markdown
Synthesis is blueprint → LLM rewrite → deterministic validator → offline-only synthesis judge → registered-tool reference execution.
```

with:

```markdown
Synthesis is blueprint → LLM rewrite → deterministic validator → LLM judge used only during data synthesis → registered-tool reference execution.
```

Add this local synthesis note near the data synthesis commands:

```markdown
Local synthesis is CPU/lightweight orchestration around an OpenAI-compatible API. It requires `openai`, `PyYAML`, `DEEPSEEK_API_KEY`, network access to the configured base URL, and a copied `config_synthesis.yaml`; it does not require a local GPU and may incur API charges.
```

- [ ] **Step 4: Update data README wording**

In `project/try_ticket_agent/data/README.md`, ensure the same sentence appears:

```markdown
The synthesis judge is an LLM judge used only during data synthesis; training and evaluation consume frozen JSONL data and never synthesize online.
```

- [ ] **Step 5: Run remote config tests**

Run:

```powershell
conda run -n all-in-rag python -m unittest project.try_ticket_agent.tests.test_remote_configs -q
```

Expected: all remote config tests pass.

- [ ] **Step 6: Commit Task 6**

Run:

```powershell
git add project/try_ticket_agent/README.md project/try_ticket_agent/data/README.md project/try_ticket_agent/tests/test_remote_configs.py
git commit -m "docs(ticket-agent): clarify synthesis judge scope"
```

---

## Final Verification

- [ ] **Step 1: Run all Ticket tests**

Run:

```powershell
conda run -n all-in-rag python -m unittest discover -s project/try_ticket_agent/tests -p "test_*.py" -q
```

Expected: all Ticket tests pass.

- [ ] **Step 2: Run shared GSPO exact-token and AgentFlow suites**

Run:

```powershell
conda run -n all-in-rag python -m unittest project.try_gsm8k_0522.tests.test_gspo_objective project.try_gsm8k_0522.tests.test_flowgrpo_light project.try_gsm8k_0522.tests.test_flowgrpo_light_agentflow -q
```

Expected: all listed GSPO tests pass.

- [ ] **Step 3: Check formatting-sensitive diffs**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 4: Inspect final status**

Run:

```powershell
git status --short --branch
```

Expected: branch `fix/ticket-agent-hardening` with a clean worktree after commits.

- [ ] **Step 5: Summarize local synthesis feasibility**

Report that local synthesis can run only after `project/try_ticket_agent/config_synthesis.yaml` is created from the example config and network/API usage is explicitly accepted. Do not run synthesis as part of tests or hardening verification.

