from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("text", "expected"),
    [("19", "19"), ("work\n#### 1,250", "1250"), ("answer -3.5", "-3.5"), ("2/4", "2/4")],
)
def test_extract_numeric_answer(text: str, expected: str) -> None:
    from agentflow_rl.tasks.gsm8k.verifier import extract_numeric_answer

    assert extract_numeric_answer(text) == expected


def test_answers_match_exact_fraction_and_decimal() -> None:
    from agentflow_rl.tasks.gsm8k.verifier import answers_match

    assert answers_match("0.5", "1/2")
    assert answers_match("1,250", "1250")
    assert not answers_match("0.51", "1/2")
    assert answers_match("0.3333333", "1/3")


def test_verifier_conclusion_uses_last_explicit_marker() -> None:
    from agentflow_rl.tasks.gsm8k.verifier import extract_verifier_conclusion

    assert extract_verifier_conclusion("Conclusion: STOP\nCorrection\nConclusion: CONTINUE") == "CONTINUE"
    assert extract_verifier_conclusion("Conclusion: STOP") == "STOP"
    with pytest.raises(ValueError):
        extract_verifier_conclusion("looks complete")
