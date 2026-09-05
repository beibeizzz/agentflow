from __future__ import annotations

import json

import pytest

from agentflow_rl.runtime.errors import ActionParseError
from agentflow_rl.tasks.deepresearch.dataset import (
    allows_retrieval_stage_reuse,
    context_documents,
    deduplicate_documents,
    deterministic_subset,
    source_context_corpora,
    standardize_example,
)
from agentflow_rl.tasks.deepresearch.eval_split import split_labeled_rows
from agentflow_rl.tasks.deepresearch.corpus import normalize as normalize_corpus_row, rows as corpus_rows
from agentflow_rl.tasks.deepresearch.retrieval import (
    InMemoryBM25Index,
    ResearchDocument,
    validate_supporting_fact_coverage,
)
from agentflow_rl.tasks.deepresearch.schemas import DeepResearchExample, ResearchAction, ResearchFinalAnswer
from agentflow_rl.tasks.deepresearch.tools import DeepResearchEnvironment
from agentflow_rl.tasks.deepresearch.verifier import evaluate_research_answer


def test_bm25_search_and_read_are_deterministic() -> None:
    documents = [
        ResearchDocument("a", "Paris", ("Paris is the capital of France.",)),
        ResearchDocument("b", "Berlin", ("Berlin is the capital of Germany.",)),
    ]
    index = InMemoryBM25Index(documents)

    hits = index.search("capital France", top_k=1)

    assert hits[0].doc_id == "a"
    assert index.read("a").sentences[0].startswith("Paris")


def test_research_action_accepts_qwen_empty_think_and_json_fence() -> None:
    action = ResearchAction.parse(
        '<think>\n\n</think>\n```json\n'
        '{"sub_goal":"search","tool_name":"Research_Search_Tool","arguments":{"query":"Paris"}}\n'
        '```'
    )

    assert action.tool_name == "Research_Search_Tool"


def test_research_action_rejects_removed_finish_tool() -> None:
    with pytest.raises(ActionParseError):
        ResearchAction.parse(
            '{"sub_goal":"finish","tool_name":"Research_Finish_Tool","arguments":{}}'
        )


def test_research_final_answer_accepts_qwen_empty_think_and_json_fence() -> None:
    answer = ResearchFinalAnswer.parse(
        '<think>\n\n</think>\n```json\n'
        '{"answer":"Paris","report":"Evidence.",'
        '"citations":[{"title":"Paris","sentence_id":0}]}\n```'
    )

    assert answer.answer == "Paris"


def test_research_read_preserves_sentence_ids_and_paginates() -> None:
    index = InMemoryBM25Index([
        ResearchDocument("doc", "Doc", tuple(f"sentence {index}" for index in range(45))),
    ])
    environment = DeepResearchEnvironment(index)
    action = ResearchAction.model_validate({
        "sub_goal": "read next page",
        "tool_name": "Research_Read_Tool",
        "arguments": {"doc_id": "doc", "start_sentence": 20, "max_sentences": 20},
    })

    result = environment.execute(action)

    assert result["data"]["sentences"][0]["sentence_id"] == 20
    assert result["data"]["next_start_sentence"] == 40
    assert result["data"]["total_sentences"] == 45


def test_hotpot_joint_reward_uses_answer_and_supporting_facts() -> None:
    example = DeepResearchExample.from_row({
        "_id": "q1",
        "dataset": "hotpotqa",
        "question": "What is the capital?",
        "answer": "Paris",
        "supporting_facts": {"title": ["Paris"], "sent_id": [0]},
    })
    prediction = ResearchFinalAnswer.model_validate({
        "answer": "Paris",
        "report": "Paris is the capital.",
        "citations": [{"title": "Paris", "sentence_id": 0}],
    })

    result = evaluate_research_answer(prediction, example)

    assert result.success is True
    assert result.reward == 1.0
    assert result.metrics["joint_f1"] == 1.0


def test_research_verifier_reports_citation_grounding_without_changing_reward() -> None:
    from agentflow_rl.tasks.deepresearch.schemas import Citation

    example = DeepResearchExample.from_row({
        "_id": "q-grounding",
        "dataset": "hotpotqa",
        "question": "What is the capital?",
        "answer": "Paris",
        "supporting_facts": [["Paris", 0]],
    })
    prediction = ResearchFinalAnswer.model_validate({
        "answer": "Paris",
        "report": "Paris is the capital.",
        "citations": [{"title": "Paris", "sentence_id": 0}],
    })

    result = evaluate_research_answer(
        prediction,
        example,
        observed_citations=[Citation(title="Berlin", sentence_id=0)],
    )

    assert result.reward == 1.0
    assert result.metrics["citation_grounded_fraction"] == 0.0
    assert result.metrics["citation_grounded_exact"] == 0.0


def test_context_corpus_prefers_the_more_complete_title_record() -> None:
    short = context_documents({"context": [["Paris", ["short"]]]})
    long = context_documents({"context": {"title": [" paris "], "sentences": [["long", "record"]]}})

    result = deduplicate_documents([*short, *long])

    assert len(result) == 1
    assert result[0].sentences == ("long", "record")


def test_standardized_distractor_row_carries_its_local_retrieval_documents() -> None:
    row = {
        "_id": "q-local",
        "question": "Where?",
        "answer": "Paris",
        "supporting_facts": [["Paris", 0]],
        "context": [["Paris", ["Paris is in France."]]],
    }

    result = standardize_example(row, dataset="hotpotqa")

    assert result["metadata"]["retrieval_documents"][0]["title"] == "Paris"


def test_same_hotpot_example_can_progress_from_distractor_to_fullwiki() -> None:
    example = {
        "episode_id": "hotpot-1",
        "dataset": "hotpotqa",
        "question": "Where?",
        "answer": "Paris",
        "supporting_facts": [{"title": "Paris", "sentence_id": 0}],
    }

    assert allows_retrieval_stage_reuse(
        "hotpot_distractor", "hotpot_fullwiki", example, dict(example)
    )
    assert not allows_retrieval_stage_reuse(
        "hotpot_distractor", "validation", example, dict(example)
    )


def test_deepresearch_subset_is_order_independent() -> None:
    rows = [{"episode_id": str(index)} for index in range(20)]

    assert deterministic_subset(rows, 5) == deterministic_subset(list(reversed(rows)), 5)


def test_research_backend_coverage_checks_title_and_sentence_alignment() -> None:
    index = InMemoryBM25Index([
        ResearchDocument("paris", "Paris", ("Paris is in France.", "It is a capital.")),
    ])
    example = DeepResearchExample.from_row({
        "_id": "coverage",
        "dataset": "hotpotqa",
        "question": "Where?",
        "answer": "Paris",
        "supporting_facts": [["Paris", 1]],
    })

    assert validate_supporting_fact_coverage(index, [example]) == 1


def test_research_backend_rejects_sentence_content_mismatch() -> None:
    index = InMemoryBM25Index([
        ResearchDocument("paris", "Paris", ("Different sentence.",)),
    ])
    example = DeepResearchExample.from_row({
        "_id": "coverage-mismatch",
        "dataset": "hotpotqa",
        "question": "Where?",
        "answer": "Paris",
        "supporting_facts": [["Paris", 0]],
        "metadata": {
            "retrieval_documents": [{
                "doc_id": "source-paris",
                "title": "Paris",
                "sentences": ["Paris is in France."],
            }],
        },
    })

    with pytest.raises(RuntimeError, match="differs from benchmark context"):
        validate_supporting_fact_coverage(index, [example])


def test_labeled_eval_split_is_disjoint_and_order_independent() -> None:
    rows = [{"_id": str(index), "question": f"Question {index}"} for index in range(20)]

    first = split_labeled_rows(rows, validation_size=4, test_size=8)
    second = split_labeled_rows(reversed(rows), validation_size=4, test_size=8)

    assert first == second
    assert {row["_id"] for row in first[0]}.isdisjoint(
        row["_id"] for row in first[1]
    )


def test_source_context_corpora_are_separated_by_benchmark(tmp_path) -> None:
    counts = source_context_corpora({
        "hotpotqa": [ResearchDocument("h", "Hotpot", ("Hotpot fact.",))],
        "2wiki": [ResearchDocument("w", "Two Wiki", ("TwoWiki fact.",))],
    }, tmp_path)

    hotpot = json.loads((tmp_path / "hotpotqa" / "documents.jsonl").read_text())
    two_wiki = json.loads((tmp_path / "2wiki" / "documents.jsonl").read_text())
    assert counts == {"2wiki": 1, "hotpotqa": 1}
    assert hotpot["title"] == "Hotpot"
    assert hotpot["contents"] == "Hotpot Hotpot fact."
    assert two_wiki["title"] == "Two Wiki"


def test_hotpot_corpus_converter_uses_first_nonempty_sentence_paragraph() -> None:
    result = normalize_corpus_row({
        "id": "42",
        "title": "Paris",
        "text": [[], ["Sentence zero.", "Sentence one."], ["Later paragraph."]],
    })

    assert result["sentences"] == ["Sentence zero.", "Sentence one."]


def test_corpus_converter_reads_bzip2_shards_from_directory(tmp_path) -> None:
    import bz2
    import json

    shard = tmp_path / "AA" / "wiki_00.bz2"
    shard.parent.mkdir()
    with bz2.open(shard, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "1", "title": "One", "text": [["Fact."]]}) + "\n")

    loaded = list(corpus_rows(tmp_path))

    assert loaded[0]["title"] == "One"
