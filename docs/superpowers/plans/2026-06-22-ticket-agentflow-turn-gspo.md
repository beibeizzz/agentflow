# Ticket AgentFlow and Turn-Level GSPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a core-AgentFlow Ticket baseline and a matching no-SFT Qwen3-0.6B LoRA turn-level GSPO experiment using three deterministic tools and binary reward.

**Architecture:** Extend core AgentFlow with opt-in structured Planner/Executor modes and workflow-only output while preserving every legacy default. Register stateful Ticket BaseTools, bind one isolated backend per Solver, synthesize requests from deterministic blueprints through a validated DeepSeek-compatible rewrite pipeline, then reuse the existing exact-token AgentFlow rollout and turn-level GSPO objective.

**Tech Stack:** Python 3.10+, AgentFlow, Pydantic, dataclasses, OpenAI-compatible SDK, PyTorch, Transformers, PEFT, PyYAML, unittest, Bash, Qwen3-0.6B.

---

## Working-Tree Safety

The workspace already contains uncommitted GSPO changes in `project/try_gsm8k_0522`. Preserve them. Do not create a clean worktree from `HEAD`, because it would omit the exact-token implementation this task depends on. Never run `git add -A`; every commit below stages exact paths only.

Before implementation, capture:

```bash
git status --short
git diff -- project/try_gsm8k_0522/flowgrpo_light/agentflow_rollout.py
```

The second command is the base for the later incremental callback extension.

## Planned Files

```text
project/agentflow/agentflow/models/{formatters.py,planner.py,executor.py}
project/agentflow/agentflow/solver.py
project/agentflow/agentflow/tools/ticket_common/
project/agentflow/agentflow/tools/ticket_query/
project/agentflow/agentflow/tools/ticket_update/
project/agentflow/agentflow/tools/ticket_finish/
project/try_ticket_agent/tests/
project/try_gsm8k_0522/flowgrpo_light/agentflow_rollout.py
project/try_gsm8k_0522/tests/test_flowgrpo_light_agentflow.py
project/try_ticket_agent/
```

### Task 1: Add strict structured Planner actions without changing legacy prompts

**Files:**
- Modify: `project/agentflow/agentflow/models/formatters.py`
- Modify: `project/agentflow/agentflow/models/planner.py`
- Create: `project/try_ticket_agent/__init__.py`
- Create: `project/try_ticket_agent/tests/__init__.py`
- Create: `project/try_ticket_agent/tests/test_agentflow_structured_workflow.py`

- [ ] **Step 0: Bootstrap the inner AgentFlow source package for tests**

```python
# try_ticket_agent/tests/__init__.py
from pathlib import Path
import sys
import types

PROJECT_DIR = Path(__file__).resolve().parents[2]
AGENTFLOW_CORE = PROJECT_DIR / 'agentflow' / 'agentflow'
agentflow_pkg = types.ModuleType('agentflow')
agentflow_pkg.__path__ = [str(AGENTFLOW_CORE)]
agentflow_pkg.__file__ = str(AGENTFLOW_CORE / '__init__.py')
sys.modules['agentflow'] = agentflow_pkg
```

- [ ] **Step 1: Write failing formatter and extraction tests**

```python
class StructuredPlannerTests(unittest.TestCase):
    def test_structured_action_accepts_tool_and_arguments(self):
        action = StructuredToolAction(
            tool_name='Ticket_Update_Tool',
            arguments={'ticket_id': 'T-1', 'field': 'priority', 'value': 'urgent'},
        )
        self.assertEqual(action.tool_name, 'Ticket_Update_Tool')

    def test_structured_extraction_rejects_wrappers_and_extra_keys(self):
        planner = make_planner_without_init(action_mode='structured')
        invalid = [
            '```json\n{"tool_name":"Ticket_Finish_Tool","arguments":{}}\n```',
            '{"tool_name":"Ticket_Finish_Tool","arguments":{},"extra":1}',
            '[{"tool_name":"Ticket_Finish_Tool","arguments":{}}]',
        ]
        for response in invalid:
            self.assertEqual(planner.extract_context_subgoal_and_tool(response), (None, None, None))
```

Snapshot the existing Calculator-only prompt branch and verify default `action_mode` produces the same prompt and response-format class.

- [ ] **Step 2: Run the red test**

Run from `project/`:

```bash
conda run -n all-in-rag python -m unittest try_ticket_agent.tests.test_agentflow_structured_workflow -v
```

Expected: import failure for `StructuredToolAction` or unexpected `action_mode`.

- [ ] **Step 3: Implement the formatter and opt-in Planner branch**

```python
class StructuredToolAction(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    model_config = ConfigDict(extra='forbid')
```

Add `action_mode: str = 'legacy'` to `Planner.__init__`, validate `{'legacy', 'structured'}`, and store it. Add a structured branch before existing prompt branches; require one JSON object, list `available_tools` and `toolbox_metadata`, and use `StructuredToolAction` as `response_format`.

Add `_extract_structured_action(response)` that accepts a Pydantic instance or raw `json.loads` object, requires exactly `tool_name,arguments`, rejects wrappers/prose, and returns:

```python
(
    json.dumps(action.arguments, sort_keys=True, separators=(',', ':')),
    f'Execute {action.tool_name}',
    action.tool_name,
)
```

Call it only in structured mode; leave legacy extraction unchanged.

- [ ] **Step 4: Run focused and existing Planner tests**

```bash
conda run -n all-in-rag python -m unittest try_ticket_agent.tests.test_agentflow_structured_workflow try_gsm8k_0522.tests.test_role_prompts -v
```

- [ ] **Step 5: Commit exact files**

```bash
git add project/agentflow/agentflow/models/formatters.py project/agentflow/agentflow/models/planner.py project/try_ticket_agent/__init__.py project/try_ticket_agent/tests/__init__.py project/try_ticket_agent/tests/test_agentflow_structured_workflow.py
git commit -m "feat(agentflow): add structured planner actions"
```

### Task 2: Add deterministic structured Executor mode

**Files:**
- Modify: `project/agentflow/agentflow/models/executor.py`
- Modify: `project/try_ticket_agent/tests/test_agentflow_structured_workflow.py`

- [ ] **Step 1: Write failing no-LLM/direct-dispatch tests**

```python
def test_structured_executor_calls_cached_tool_directly():
    tool = RecordingTool()
    executor = Executor.__new__(Executor)
    executor.execution_mode = 'structured'
    executor.tool_instances_cache = {'Ticket_Update_Tool': tool}
    result = executor.execute_tool_command(
        'Ticket_Update_Tool',
        '{"field":"priority","ticket_id":"T-1","value":"urgent"}',
    )
    assert result == {'ok': True}
    assert tool.calls == [{'field': 'priority', 'ticket_id': 'T-1', 'value': 'urgent'}]

def test_structured_generate_command_does_not_call_llm():
    executor = make_executor(execution_mode='structured', llm=FailIfCalled())
    command = executor.generate_tool_command('', None, '{"ticket_id":"T-1"}', '', 'Ticket_Finish_Tool', {}, 1)
    assert command.command == '{"ticket_id":"T-1"}'
```

Add invalid JSON, non-object arguments, unknown tool, structured tool exception, and legacy Calculator assertions.

- [ ] **Step 2: Run red tests**

Expected: constructor rejects the mode or enters legacy LLM/exec code.

- [ ] **Step 3: Implement structured Executor**

Add `execution_mode: str = 'legacy'`. In structured mode do not construct `llm_generate_tool_command`; `set_query_cache_dir` must not create directories. `generate_tool_command` returns:

```python
ToolCommand(
    analysis='Deterministic structured dispatch.',
    explanation='Arguments are passed directly to the registered tool.',
    command=context,
)
```

At the start of `execute_tool_command`, parse one JSON object, fetch the exact cached tool, call `execute(**arguments)`, and convert exceptions to `{'ok': False, 'code': 'TOOL_EXECUTION_ERROR', 'message': str(exc)}`. Return before legacy command splitting and `exec`.

- [ ] **Step 4: Run Executor regressions**

```bash
conda run -n all-in-rag python -m unittest try_ticket_agent.tests.test_agentflow_structured_workflow try_gsm8k_0522.tests.test_executor_calculator_prompt -v
```

- [ ] **Step 5: Commit**

```bash
git add project/agentflow/agentflow/models/executor.py project/try_ticket_agent/tests/test_agentflow_structured_workflow.py
git commit -m "feat(agentflow): add deterministic structured executor"
```

### Task 3: Add workflow-only Solver output and constructor wiring

**Files:**
- Modify: `project/agentflow/agentflow/solver.py`
- Modify: `project/try_ticket_agent/tests/test_agentflow_structured_workflow.py`

- [ ] **Step 1: Write failing Solver tests**

```python
def test_workflow_runs_actions_without_final_generators():
    planner = ScriptedPlanner()
    solver = Solver(planner, StopVerifier(), Memory(), RecordingExecutor(), output_types='workflow', max_steps=1)
    result = solver.solve('update ticket')
    assert result['step_count'] == 1
    assert 'direct_output' not in result
    assert 'final_output' not in result
    assert planner.final_calls == 0

def test_construct_solver_defaults_remain_legacy():
    signature = inspect.signature(construct_solver)
    assert signature.parameters['planner_action_mode'].default == 'legacy'
    assert signature.parameters['executor_mode'].default == 'legacy'
```

- [ ] **Step 2: Run red tests**

Expected: `workflow` fails validation or does not enter the action loop.

- [ ] **Step 3: Wire opt-in modes**

Allow `workflow`, include it in the query/action condition, and add no final-output branch for it. Add `planner_action_mode='legacy'` and `executor_mode='legacy'` to `construct_solver` and pass them to Planner/Executor.

- [ ] **Step 4: Run core and GSM8K Solver tests**

```bash
conda run -n all-in-rag python -m unittest try_ticket_agent.tests.test_agentflow_structured_workflow try_gsm8k_0522.tests.test_gsm8k_scripts -v
```

- [ ] **Step 5: Commit**

```bash
git add project/agentflow/agentflow/solver.py project/try_ticket_agent/tests/test_agentflow_structured_workflow.py
git commit -m "feat(agentflow): add workflow-only solver mode"
```

### Task 4: Add Ticket schemas, backend, and registered BaseTools

**Files:**
- Create: `project/agentflow/agentflow/tools/ticket_common/{__init__.py,schemas.py,backend.py}`
- Create: `project/agentflow/agentflow/tools/ticket_query/{__init__.py,tool.py}`
- Create: `project/agentflow/agentflow/tools/ticket_update/{__init__.py,tool.py}`
- Create: `project/agentflow/agentflow/tools/ticket_finish/{__init__.py,tool.py}`
- Create: `project/try_ticket_agent/tests/test_ticket_tools.py`

- [ ] **Step 1: Write backend/tool red tests**

```python
def test_three_tools_share_bound_backend_and_update_atomically():
    backend = TicketBackend(); backend.reset(make_state(), make_goal())
    query = Ticket_Query_Tool(); update = Ticket_Update_Tool(); finish = Ticket_Finish_Tool()
    for tool in (query, update, finish):
        tool.bind_backend(backend)
    assert query.execute(lookup_by='ticket_id', value='T-1')['data']['ticket_id'] == 'T-1'
    assert update.execute(ticket_id='T-1', field='priority', value='urgent')['ok'] is True
    assert finish.execute(ticket_id='T-1', outcome='completed')['ok'] is True

def test_rejected_transition_does_not_mutate_state():
    backend = ready_backend(status='resolved')
    result = backend.update('T-1', 'status', 'open')
    assert result['code'] == 'ILLEGAL_TRANSITION'
    assert backend.tickets['T-1'].status == 'resolved'
```

Cover unbound backend, missing ticket, invalid lookup/field/enum, closed ticket, non-unique query, finish immutability, reset deep copy, state diff, and JSON serialization.

- [ ] **Step 2: Run red tests**

```bash
conda run -n all-in-rag python -m unittest try_ticket_agent.tests.test_ticket_tools -v
```

- [ ] **Step 3: Implement strict schemas and backend**

Define `Ticket`, `GoalSpec`, `FinishSubmission`, and `ActionEvent` dataclasses. Centralize legal status transitions. `TicketBackend.reset(initial_state, goal_spec)` deep-copies an immutable baseline and working state. Operations append events and return `{ok,code,message,data}`.

Each tool sets complete BaseTool metadata, stores `backend: TicketBackend | None`, exposes `bind_backend`, and delegates `execute`. Use exact class/tool names required by Initializer.

- [ ] **Step 4: Run tests and Initializer discovery assertions**

Expected: all three tool classes are discovered and state tests pass.

- [ ] **Step 5: Commit**

```bash
git add project/agentflow/agentflow/tools/ticket_common project/agentflow/agentflow/tools/ticket_query project/agentflow/agentflow/tools/ticket_update project/agentflow/agentflow/tools/ticket_finish project/try_ticket_agent/tests/test_ticket_tools.py
git commit -m "feat(agentflow): add sandbox ticket tools"
```

### Task 5: Build Ticket Verifier, Solver factory, and AgentFlow baseline

**Files:**
- Create: `project/try_ticket_agent/ticket_env/{__init__.py,episode_io.py,verifier.py,solver_factory.py}`
- Create: `project/try_ticket_agent/run_ticket_agentflow.py`
- Create: `project/try_ticket_agent/tests/test_ticket_solver.py`

- [ ] **Step 1: Write fake-engine end-to-end red tests**

```python
def test_direct_episode_uses_core_solver_and_two_planner_turns():
    runtime = build_test_runtime([update_action, finish_action])
    result = runtime.run(make_direct_episode())
    assert result['verification']['success'] is True
    assert result['reward'] == 1.0
    assert result['step_count'] == 2
    assert runtime.solver.__class__.__module__ == 'agentflow.solver'

def test_hidden_goal_never_enters_prompts_or_memory():
    runtime = build_test_runtime([update_action, finish_action])
    result = runtime.run(make_direct_episode(secret_marker='HIDDEN-991'))
    visible = json.dumps(result['planner_prompts']) + json.dumps(result['memory'])
    assert 'HIDDEN-991' not in visible
```

Add indirect three-turn success, wrong finish, invalid JSON, tool error, collateral mutation, reset between episodes, and repeated IDs across runtimes.

- [ ] **Step 2: Run red tests**

```bash
conda run -n all-in-rag python -m unittest try_ticket_agent.tests.test_ticket_solver -v
```

- [ ] **Step 3: Implement verifier and factory**

`TicketVerifier.verificate_context` reads backend and Memory and returns `MemoryVerification(analysis='Deterministic ticket state check.', stop_signal=backend.finish_submission is not None)`. `extract_conclusion` maps to `STOP/CONTINUE`. `verify_final` requires exact goal, finish, no collateral diff, invalid action, tool error, or step overflow.

`construct_ticket_solver` calls core `construct_solver` with the three tools, `planner_action_mode='structured'`, `executor_mode='structured'`, and `output_types='workflow'`; binds one backend to cached tools; replaces the generic verifier; and resets Memory/backend per row.

`run_ticket_agentflow.py` follows the GSM8K runner: bootstrap source AgentFlow, check `/models`, load rows, reset each episode, atomically write results, support resume/overwrite, and report binary metrics.

- [ ] **Step 4: Run Ticket baseline tests**

- [ ] **Step 5: Commit**

```bash
git add project/try_ticket_agent/ticket_env project/try_ticket_agent/run_ticket_agentflow.py project/try_ticket_agent/tests/test_ticket_solver.py
git commit -m "feat(ticket-agent): add core AgentFlow baseline"
```

### Task 6: Generate deterministic blueprints and reference-validate them

**Files:**
- Create: `project/try_ticket_agent/data_synthesis/{__init__.py,schemas.py,blueprints.py}`
- Create: `project/try_ticket_agent/scripts/{generate_blueprints.py,validate_dataset.py}`
- Create: `project/try_ticket_agent/tests/test_blueprints.py`

- [ ] **Step 1: Write determinism/curriculum red tests**

```python
def test_blueprints_are_deterministic_and_exactly_eighty_twenty():
    left = [generate_blueprint(seed=42, split='train', index=i) for i in range(100)]
    right = [generate_blueprint(seed=42, split='train', index=i) for i in range(100)]
    assert encode_blueprints(left) == encode_blueprints(right)
    assert Counter(item.lookup_mode for item in left) == {'ticket_id': 80, 'indirect': 20}

def test_reference_actions_pass_registered_tools_and_verifier():
    result = execute_reference_actions(generate_blueprint(seed=42, split='train', index=4))
    assert result.success is True
```

Also assert 6-10 tickets, unique indirect keys, one legal update, no indirect target ticket ID, and cross-split ID/state/signature isolation.

- [ ] **Step 2: Run red tests**

- [ ] **Step 3: Implement deterministic blueprints**

Use `random.Random(stable_integer_hash(generator_version, seed, split, index))`. Make `index % 5 == 4` indirect and alternate customer/order lookup. Build one canonical public request. Execute reference Query if required, Update, and Finish through structured Executor and registered tools.

- [ ] **Step 4: Generate and validate blueprint fixtures**

```bash
conda run -n all-in-rag python try_ticket_agent/scripts/generate_blueprints.py --seed 42 --smoke 32 --train 2500 --validation 256 --test 512 --output-dir try_ticket_agent/data/blueprints
conda run -n all-in-rag python try_ticket_agent/scripts/validate_dataset.py --blueprints try_ticket_agent/data/blueprints
```

Expected: all references pass, train mix is 2000/500, and no split collision occurs.

- [ ] **Step 5: Commit**

```bash
git add project/try_ticket_agent/data_synthesis project/try_ticket_agent/scripts project/try_ticket_agent/tests/test_blueprints.py project/try_ticket_agent/data/blueprints
git commit -m "feat(ticket-data): add deterministic episode blueprints"
```

### Task 7: Add DeepSeek-compatible rewrite and judge synthesis

**Files:**
- Create: `project/try_ticket_agent/data_synthesis/{prompts.py,api_client.py,validators.py,pipeline.py}`
- Create: `project/try_ticket_agent/scripts/synthesize_dataset.py`
- Create: `project/try_ticket_agent/config_synthesis.example.yaml`
- Create: `project/try_ticket_agent/tests/{test_synthesis_client.py,test_synthesis_pipeline.py}`

- [ ] **Step 1: Write fake-SDK red tests**

```python
def test_validator_rejects_indirect_ticket_leak_and_second_update():
    blueprint = indirect_blueprint()
    leaked = f'For ticket {blueprint.target_ticket_id}, set priority urgent and status resolved.'
    result = validate_candidate(blueprint, leaked)
    assert {'target_ticket_leak', 'multiple_mutations'} <= set(result.codes)

def test_pipeline_retries_with_feedback_and_resumes(tmp_path):
    client = FakeClient(rewrites=[bad_candidate, good_candidate], judges=[accepted_judge])
    pipeline = TicketSynthesisPipeline(client, max_attempts=3)
    first = pipeline.run([direct_blueprint()], paths=PipelinePaths.in_directory(tmp_path), resume=True)
    second = pipeline.run([direct_blueprint()], paths=PipelinePaths.in_directory(tmp_path), resume=True)
    assert first['accepted'] == 1
    assert second['api_calls'] == 0
```

Cover empty/non-object JSON, truncation, transport backoff, judge schema/rejection, concurrency ordering, usage totals, rejected records, and atomic writes.

- [ ] **Step 2: Run red tests without network**

- [ ] **Step 3: Implement API and synthesis pipeline**

Inject `sdk_client`, request `response_format={'type':'json_object'}`, separate transport and semantic attempts, and read only `DEEPSEEK_API_KEY`. Rewrite schema is `{'user_request': str}`; judge schema is `{'accepted': bool, 'reasons': list[str]}`.

Validate required values, introduced IDs/enums, indirect leakage, second operations, tool/action/goal hints, uniqueness, length, and characters. After judge acceptance, reference-verify and store request plus blueprint state/goal without reference actions.

- [ ] **Step 4: Run synthesis tests**

```bash
conda run -n all-in-rag python -m unittest try_ticket_agent.tests.test_synthesis_client try_ticket_agent.tests.test_synthesis_pipeline -v
```

- [ ] **Step 5: Commit offline-tested synthesis**

```bash
git add project/try_ticket_agent/data_synthesis project/try_ticket_agent/scripts/synthesize_dataset.py project/try_ticket_agent/config_synthesis.example.yaml project/try_ticket_agent/tests/test_synthesis_client.py project/try_ticket_agent/tests/test_synthesis_pipeline.py
git commit -m "feat(ticket-data): add validated LLM request synthesis"
```

- [ ] **Step 6: Run real synthesis only with explicit credentials**

```bash
conda run -n all-in-rag python try_ticket_agent/scripts/synthesize_dataset.py --config try_ticket_agent/config_synthesis.yaml
conda run -n all-in-rag python try_ticket_agent/scripts/validate_dataset.py --dataset try_ticket_agent/data/generated
```

Expected: accepted/rejected/progress/summary files and manifest hashes. Without credentials, stop with `DEEPSEEK_API_KEY is not set`; never substitute template data silently.

### Task 8: Generalize the existing exact-token AgentFlow rollout incrementally

**Files:**
- Modify: `project/try_gsm8k_0522/flowgrpo_light/agentflow_rollout.py`
- Modify: `project/try_gsm8k_0522/tests/test_flowgrpo_light_agentflow.py`
- Create: `project/try_ticket_agent/tests/test_ticket_rollout.py`

- [ ] **Step 1: Re-read the dirty diff and write callback red tests**

```python
def test_generic_adapter_receives_solver_result_and_exact_samples():
    adapter = RecordingAdapter()
    runner = AgentFlowBatchRolloutRunner(
        policy=FakePolicyWithTokenIds(),
        solver_factory=solver_factory,
        reset_solver=reset_with_row,
        result_adapter=adapter,
        question_getter=lambda row: row['user_request'],
        rollout_concurrency=2,
        planner_batch_size=2,
    )
    groups = runner.run_batch([episode_row()], group_size=2)
    assert groups[0][0].samples[0].response_token_ids == [201, 202]
    assert adapter.calls == 2
```

Retain all GSM8K batch/token tests and add a test that omitting new callbacks uses the original reward behavior.

- [ ] **Step 2: Run red tests**

- [ ] **Step 3: Add optional hooks without rewriting token logic**

```python
QuestionGetter = Callable[[dict[str, Any]], str]
ResetSolver = Callable[[Any, dict[str, Any]], None]
ResultAdapter = Callable[[Any, dict[str, Any], dict[str, Any], list[PlannerSample]], RolloutResult]
```

Use a compatibility wrapper for the existing one-argument reset. Worker order is reset with row, clear samples, solve selected question, call custom adapter when present, otherwise execute unchanged GSM reward extraction. Do not alter `_planner_sample`, prompt rendering, or token storage.

- [ ] **Step 4: Run shared and Ticket rollout tests**

```bash
conda run -n all-in-rag python -m unittest try_gsm8k_0522.tests.test_flowgrpo_light_agentflow try_ticket_agent.tests.test_ticket_rollout -v
```

- [ ] **Step 5: Commit incremental shared changes**

```bash
git add project/try_gsm8k_0522/flowgrpo_light/agentflow_rollout.py project/try_gsm8k_0522/tests/test_flowgrpo_light_agentflow.py project/try_ticket_agent/tests/test_ticket_rollout.py
git commit -m "feat(flowgrpo): support task-specific AgentFlow rollouts"
```

### Task 9: Add Ticket turn-level GSPO training and assertions

**Files:**
- Create: `project/try_ticket_agent/flowgrpo_general_2x40g/train_ticket_gspo.py`
- Create: `project/try_ticket_agent/tests/test_ticket_gspo.py`

- [ ] **Step 1: Write grouping/turn-flattening red tests**

```python
def test_binary_group_advantage_is_broadcast_after_trajectory_normalization():
    success = rollout(reward=1.0, turns=2)
    failure = rollout(reward=0.0, turns=3)
    flat, advantages, rewards, grouped = flatten_rollout_groups([[success, failure]])
    assert rewards == [[1.0, 0.0]]
    assert advantages == [1.0, -1.0]
    items = build_loss_items(FakePolicy(), flat, advantages)
    assert [item.advantage for item in items] == [1.0, 1.0, -1.0, -1.0, -1.0]
```

Add infrastructure exclusion before mean/std, all-equal zero advantage/no update, exact-ID logprob preference, clip `(0.001, 0.003)`, and FP32 ratio assertions. Import existing objective functions.

- [ ] **Step 2: Run red tests**

- [ ] **Step 3: Implement Ticket training orchestration**

Follow `train_light_grpo_general.py`, but use Ticket factory, callbacks, and `user_request`. Import `PlannerPolicy`, `flatten_rollout_groups`, `train_step_grpo`, and helpers. Log reward groups, valid/invalid counts, zero-variance fraction, nonzero trajectories/turns, token counts, ratio/clip/KL, time, and GPU memory. Save adapters and close runner in `finally`. Expose no SFT or trajectory objective.

- [ ] **Step 4: Run Ticket/existing GSPO tests**

```bash
conda run -n all-in-rag python -m unittest try_ticket_agent.tests.test_ticket_gspo try_gsm8k_0522.tests.test_gspo_objective -v
```

- [ ] **Step 5: Commit**

```bash
git add project/try_ticket_agent/flowgrpo_general_2x40g/train_ticket_gspo.py project/try_ticket_agent/tests/test_ticket_gspo.py
git commit -m "feat(ticket-gspo): add binary turn-level training"
```

### Task 10: Add controlled evaluation and aligned remote scripts

**Files:**
- Create: `project/try_ticket_agent/baseline/config_agentflow_baseline.yaml`
- Create: `project/try_ticket_agent/baseline/run_agentflow_baseline.sh`
- Create: `project/try_ticket_agent/flowgrpo_general_2x40g/eval_ticket_agent.py`
- Create: `project/try_ticket_agent/flowgrpo_general_2x40g/config_train_general_2x40g.yaml`
- Create: `project/try_ticket_agent/flowgrpo_general_2x40g/config_eval_learnable_general_2x40g.yaml`
- Create: `project/try_ticket_agent/flowgrpo_general_2x40g/run_train_general_2x40g.sh`
- Create: `project/try_ticket_agent/flowgrpo_general_2x40g/run_eval_learnable_general_2x40g.sh`
- Create: `project/try_ticket_agent/tests/{test_ticket_eval.py,test_remote_configs.py}`

- [ ] **Step 1: Write evaluation/config red tests**

```python
def test_train_config_matches_required_remote_defaults():
    cfg = load_yaml(TRAIN_CONFIG)
    assert (cfg['question_batch_size'], cfg['group_size']) == (4, 8)
    assert (cfg['rollout_concurrency'], cfg['planner_batch_size']) == (32, 32)
    assert cfg['max_steps'] == 3
    assert (cfg['clip_range_low'], cfg['clip_range_high']) == (0.001, 0.003)
    assert cfg['reward_mode'] == 'binary'

def test_summary_separates_model_and_infrastructure_failures():
    summary = summarize_results([success_result(), invalid_action_result(), infrastructure_result()])
    assert summary['episode_success_rate'] == 1 / 3
    assert summary['invalid_action_rate'] == 1 / 3
    assert summary['infrastructure_failure_rate'] == 1 / 3
```

Parse shell defaults and assert equality with YAML for batch/group/concurrency/max steps/clip/policy epochs/item count. Assert the baseline script selects the same model, dataset, max steps, structured modes, and workflow output used by GSPO evaluation. Assert no SFT key/script.

- [ ] **Step 2: Run red tests**

- [ ] **Step 3: Implement evaluation**

Support `baseline` with base vLLM Planner and `adapter` with `PlannerPolicy(adapter_path=str(adapter_path))`; use temperature zero and identical Solver/data/max steps/verifier. Write per-episode JSONL and aggregate metrics including direct/indirect splits.

- [ ] **Step 4: Add the standalone AgentFlow baseline config and launcher**

The baseline config points to the frozen accepted validation/test file, `vllm-Qwen3-0.6B`, `http://127.0.0.1:8000/v1`, temperature zero, max steps 3, Planner structured mode, Executor structured mode, and workflow output. The shell script follows `try_gsm8k_0522/run_smoke.sh`: change to `project/`, expose environment overrides, validate the data file, and invoke `try_ticket_agent/run_ticket_agentflow.py`.

- [ ] **Step 5: Add aligned GSPO YAML/shell defaults**

Use Qwen3-0.6B, question/group 4/8, concurrency/batch 32, max steps 3, BF16, LoRA 64/128, LR `2e-6`, policy epochs 2, clip `0.001/0.003`, planner `256/1.2/0.95`. Keep frozen vLLM on GPU 0 and default training `CUDA_VISIBLE_DEVICES=1`. YAML and shell defaults must match.

- [ ] **Step 6: Test baseline, evaluation/config, and shell syntax**

```bash
conda run -n all-in-rag python -m unittest try_ticket_agent.tests.test_ticket_eval try_ticket_agent.tests.test_remote_configs -v
bash -n try_ticket_agent/baseline/run_agentflow_baseline.sh
bash -n try_ticket_agent/flowgrpo_general_2x40g/run_train_general_2x40g.sh
bash -n try_ticket_agent/flowgrpo_general_2x40g/run_eval_learnable_general_2x40g.sh
```

- [ ] **Step 7: Commit**

```bash
git add project/try_ticket_agent/baseline project/try_ticket_agent/flowgrpo_general_2x40g project/try_ticket_agent/tests/test_ticket_eval.py project/try_ticket_agent/tests/test_remote_configs.py
git commit -m "feat(ticket-eval): add two-40G training and evaluation"
```

### Task 11: Documentation, full regression, and remote evidence

**Files:**
- Create: `project/try_ticket_agent/README.md`
- Create: `project/try_ticket_agent/data/README.md`
- Modify: `project/try_ticket_agent/tests/test_remote_configs.py`

- [ ] **Step 1: Add README acceptance assertions**

Require exact commands for blueprint generation, API synthesis, validation, AgentFlow baseline, GSPO smoke/train, baseline evaluation, and adapter evaluation. Require explicit no-SFT, binary reward, per-turn ratio, offline-only synthesis judge, and preserved legacy defaults statements.

- [ ] **Step 2: Run documentation test red**

- [ ] **Step 3: Write operational documentation**

Document tool schemas, state isolation, hidden-goal boundary, curriculum, credentials, hashes, exact-token path, reward/advantage/ratio levels, GPU allocation, metrics, limitations, and this smoke command:

```bash
QUESTION_BATCH_SIZE=2 GROUP_SIZE=2 ROLLOUT_CONCURRENCY=4 PLANNER_BATCH_SIZE=4 MAX_TRAIN_ITEMS=2 EPOCHS=1 SAVE_EVERY=1 bash try_ticket_agent/flowgrpo_general_2x40g/run_train_general_2x40g.sh
```

- [ ] **Step 4: Run complete local tests**

```bash
cd project
conda run -n all-in-rag python -m unittest discover -s try_ticket_agent/tests -p 'test_*.py' -v
conda run -n all-in-rag python -m unittest try_gsm8k_0522.tests.test_gspo_objective try_gsm8k_0522.tests.test_flowgrpo_light try_gsm8k_0522.tests.test_flowgrpo_light_agentflow try_gsm8k_0522.tests.test_gsm8k_scripts -v
```

Expected: relevant tests pass. The two approved missing optional modules may be reported separately and are not repaired here.

- [ ] **Step 5: Verify scope and absence of SFT**

```bash
git diff --name-only 61f8795..HEAD
rg -n 'SFT|sft' project/try_ticket_agent
```

Expected: changes match planned paths; search finds only documentation saying SFT is absent.

- [ ] **Step 6: Commit documentation**

```bash
git add project/try_ticket_agent/README.md project/try_ticket_agent/data/README.md project/try_ticket_agent/tests/test_remote_configs.py
git commit -m "docs(ticket-agent): document reproducible experiments"
```

- [ ] **Step 7: Run remote baseline and one-update smoke**

Run base AgentFlow smoke on frozen accepted episodes. Then run GSPO on GPU 1 with frozen Qwen on GPU 0. Verify exact IDs, four trajectories, finite loss/ratio/KL when advantage is nonzero, LoRA change on update, and adapter save/reload. If rewards are equal, record `skip_no_advantage` and rerun more episodes; do not call infrastructure success or loss decrease a model improvement.

## Strict TDD Order

```text
structured formatter/Planner
-> structured Executor
-> workflow Solver
-> Ticket backend/BaseTools
-> Ticket verifier/factory/baseline
-> deterministic blueprints/reference execution
-> fake-tested LLM synthesis
-> generic exact-token rollout callbacks
-> Ticket turn-level GSPO
-> controlled evaluation/remote configs
-> docs/full regression/remote smoke
```

For every task: add one focused failing behavior, run it and inspect the expected failure, implement only enough to pass, run focused plus adjacent regressions, and stage exact files. Do not start GSPO orchestration until the core baseline, hidden verifier, reference execution, and exact token-ID propagation are green.
