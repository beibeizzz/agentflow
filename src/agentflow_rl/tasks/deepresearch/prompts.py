QUERY_SYSTEM = "Analyze the multi-hop research question and identify entities and evidence needs. Output concise text."
RESEARCH_TOOL_SCHEMA = """Allowed tools:
- Research_Search_Tool: arguments={"query": "..."}
- Research_Read_Tool: arguments={"doc_id": "...", "start_sentence": 0, "max_sentences": 20}
- Base_Generator_Tool: arguments={}
- Research_Finish_Tool: arguments={}
Output exactly {"sub_goal":"...","tool_name":"...","arguments":{...}}."""
PLANNER_SYSTEM = f"Plan one research action.\n{RESEARCH_TOOL_SCHEMA}"
EXECUTOR_SYSTEM = f"Validate and concretize the proposed action while preserving tool_name.\n{RESEARCH_TOOL_SCHEMA}"
VERIFIER_SYSTEM = "Judge whether memory supports a final answer with citations. End with Conclusion: STOP or Conclusion: CONTINUE."
GENERATOR_SYSTEM = "Produce one JSON object with answer, report, and citations. Each citation has title and sentence_id."
BASE_GENERATOR_SYSTEM = "Analyze the supplied evidence for the requested sub-goal. Output concise factual notes."


def query_prompt(question: str) -> str:
    return f"Research question:\n{question}"


def planner_prompt(question: str, memory_view: str) -> str:
    return f"Question:\n{question}\n\nMemory:\n{memory_view}\n\nChoose one action that advances the research plan."


def executor_prompt(action: str, memory_view: str) -> str:
    return f"Proposed action:\n{action}\n\nAvailable memory:\n{memory_view}"


def verifier_prompt(question: str, memory_view: str) -> str:
    return f"Question:\n{question}\n\nEvidence memory:\n{memory_view}"


def generator_prompt(question: str, memory_view: str) -> str:
    return f"Question:\n{question}\n\nEvidence memory:\n{memory_view}"
