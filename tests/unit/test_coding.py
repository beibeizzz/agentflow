from __future__ import annotations

from agentflow_rl.tasks.coding.dataset import (
    deterministic_limit,
    parse_taco_tests,
    problem_fingerprint,
    question_fingerprint,
    split_verified_rows,
    standardize_taco_row,
)
from agentflow_rl.tasks.coding.sandbox import LocalPythonSandbox
from agentflow_rl.tasks.coding.schemas import CodeExample, CodeTest, FinalCode, split_tests
from agentflow_rl.tasks.coding.schemas import CodeAction
from agentflow_rl.tasks.coding.tools import CodingEnvironment
from agentflow_rl.tasks.coding.verifier import evaluate_code


def test_taco_row_is_filtered_and_split_into_public_hidden_tests() -> None:
    row = {
        "id": "sum",
        "question": "Read two integers and print their sum.",
        "difficulty": "EASY",
        "picture_num": 0,
        "input_output": {"inputs": ["1 2\n", "4 5\n"], "outputs": ["3\n", "9\n"]},
    }

    normalized = standardize_taco_row(row, public_fraction=0.5)

    assert normalized is not None
    assert len(normalized["public_tests"]) == 1
    assert len(normalized["hidden_tests"]) == 1


def test_code_action_accepts_qwen_empty_think_and_json_fence() -> None:
    action = CodeAction.parse(
        '<think>\n\n</think>\n```json\n'
        '{"sub_goal":"write","tool_name":"Code_Write_Tool","arguments":{"code":"pass"}}\n'
        '```'
    )

    assert action.tool_name == "Code_Write_Tool"


def test_final_code_accepts_qwen_empty_think_json_and_python_fence() -> None:
    json_code = FinalCode.parse(
        '<think>\n\n</think>\n```json\n{"code":"print(42)"}\n```'
    )
    fenced_code = FinalCode.parse(
        '<think>\n\n</think>\n```python\nprint(42)\n```'
    )

    assert json_code.code == "print(42)"
    assert fenced_code.code == "print(42)"


def test_local_sandbox_supports_stdio_and_function_tests() -> None:
    sandbox = LocalPythonSandbox()
    stdio = sandbox.run(
        "a, b = map(int, input().split())\nprint(a + b)\n",
        [CodeTest(stdin="2 3\n", expected_stdout="5\n")],
        timeout_s=2,
    )
    function = sandbox.run(
        "def add(a, b):\n    return a + b\n",
        [CodeTest(fn_name="add", args=[2, 3], expected=5)],
        timeout_s=2,
    )

    assert stdio.pass_rate == 1.0
    assert function.pass_rate == 1.0


def test_public_feedback_and_hidden_terminal_reward_are_separate() -> None:
    public, hidden = split_tests(
        [
            CodeTest(stdin="1 2\n", expected_stdout="3\n"),
            CodeTest(stdin="8 9\n", expected_stdout="17\n"),
        ],
        identity="sum",
        public_fraction=0.5,
    )
    example = CodeExample(
        episode_id="sum",
        question="sum",
        difficulty="EASY",
        public_tests=public,
        hidden_tests=hidden,
    )
    sandbox = LocalPythonSandbox()
    environment = CodingEnvironment(example, sandbox)
    environment.code = "a, b = map(int, input().split())\nprint(a + b)\n"

    public_result = environment.sandbox.run(environment.code, example.public_tests, timeout_s=2)
    verification, hidden_result = evaluate_code(environment.code, example, sandbox, timeout_s=2)

    assert public_result.pass_rate == 1.0
    assert hidden_result.pass_rate == 1.0
    assert verification.reward == 1.0


def test_parse_taco_function_tests_preserves_arguments() -> None:
    tests = parse_taco_tests({"fn_name": "add", "inputs": [[1, 2]], "outputs": [3]})
    assert tests[0].args == [1, 2]


def test_parse_taco_stdio_lists_become_line_oriented_text() -> None:
    tests = parse_taco_tests({"inputs": [["2", "1 2"]], "outputs": [["3", "4"]]})

    assert tests[0].stdin == "2\n1 2"
    assert tests[0].expected_stdout == "3\n4"


def test_local_sandbox_supports_solution_class_wrapped_outputs_and_float_tolerance() -> None:
    sandbox = LocalPythonSandbox()
    function = sandbox.run(
        "class Solution:\n    def pair(self, value):\n        return [value, value + 1]\n",
        [CodeTest(fn_name="pair", args=[2], expected=[[2, 3]])],
        timeout_s=2,
    )
    stdio = sandbox.run(
        "print(1 / 3)\n",
        [CodeTest(stdin="", expected_stdout="0.333333")],
        timeout_s=2,
    )

    assert function.pass_rate == 1.0
    assert stdio.pass_rate == 1.0


def test_verified_source_split_is_seeded_disjoint_and_stable() -> None:
    rows = [
        {
            "episode_id": str(index),
            "metadata": {"question_fingerprint": f"fingerprint-{index}"},
        }
        for index in range(20)
    ]

    first = split_verified_rows(rows, validation_fraction=0.1, test_fraction=0.1, seed=42)
    second = split_verified_rows(list(reversed(rows)), validation_fraction=0.1, test_fraction=0.1, seed=42)

    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "train": 16,
        "validation": 2,
        "test": 2,
    }
    assert len({row["episode_id"] for values in first.values() for row in values}) == 20


def test_coding_limit_is_order_independent() -> None:
    rows = [{"episode_id": str(index)} for index in range(20)]

    assert deterministic_limit(rows, 5) == deterministic_limit(list(reversed(rows)), 5)


def test_problem_fingerprint_ignores_source_url_and_json_key_order() -> None:
    first = {
        "question": "Add two numbers",
        "url": "source-a",
        "input_output": {"inputs": ["1 2"], "outputs": ["3"]},
    }
    second = {
        "question": "  ADD   two numbers ",
        "url": "source-b",
        "input_output": '{"outputs":["3"],"inputs":["1 2"]}',
    }

    assert problem_fingerprint(first) == problem_fingerprint(second)


def test_question_fingerprint_detects_same_problem_with_different_tests() -> None:
    first = {
        "question": "Add two numbers",
        "input_output": {"inputs": ["1 2"], "outputs": ["3"]},
    }
    second = {
        "question": "  ADD   two numbers ",
        "input_output": {"inputs": ["2 3"], "outputs": ["5"]},
    }

    assert question_fingerprint(first) == question_fingerprint(second)
    assert problem_fingerprint(first) != problem_fingerprint(second)


def test_docker_sandbox_applies_container_security_and_reads_runner_result(monkeypatch) -> None:
    import json
    from types import SimpleNamespace

    from agentflow_rl.tasks.coding.sandbox import DockerSandbox

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        request = json.loads(kwargs["input"])
        assert request["code"] == "print(input())"
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"passed": 1, "total": 1, "failures": [], "timed_out": False}),
            stderr="",
        )

    monkeypatch.setattr("agentflow_rl.tasks.coding.sandbox.subprocess.run", fake_run)
    result = DockerSandbox().run(
        "print(input())",
        [CodeTest(stdin="ok\n", expected_stdout="ok\n")],
        timeout_s=2,
    )

    command = commands[0]
    assert result.pass_rate == 1.0
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert command[command.index("--memory-swap") + 1] == "4g"
    assert "-v" not in command
    assert "-i" in command


def test_docker_sandbox_rejects_malformed_backend_result(monkeypatch) -> None:
    from types import SimpleNamespace

    from agentflow_rl.tasks.coding.sandbox import DockerSandbox

    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr("agentflow_rl.tasks.coding.sandbox.subprocess.run", fake_run)

    try:
        DockerSandbox().run(
            "print(input())",
            [CodeTest(stdin="ok\n", expected_stdout="ok\n")],
            timeout_s=2,
        )
    except RuntimeError as exc:
        assert "invalid result" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
