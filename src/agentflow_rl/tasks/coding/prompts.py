QUERY_SYSTEM = "Analyze the programming problem, interface, constraints, and likely algorithm. Output concise text."
CODE_TOOL_SCHEMA = """Allowed tools:
- Base_Generator_Tool: arguments={}
- Code_Write_Tool: arguments={"code": "complete Python 3 solution"}
- Code_Run_Tests_Tool: arguments={}
- Code_Inspect_Error_Tool: arguments={}
- Code_Finish_Tool: arguments={}
Output exactly {"sub_goal":"...","tool_name":"...","arguments":{...}}."""
PLANNER_SYSTEM = f"Plan one coding action.\n{CODE_TOOL_SCHEMA}"
EXECUTOR_SYSTEM = f"Validate and concretize the proposed action while preserving tool_name.\n{CODE_TOOL_SCHEMA}"
VERIFIER_SYSTEM = "Judge whether the current code is ready for final submission. End with Conclusion: STOP or Conclusion: CONTINUE."
GENERATOR_SYSTEM = "Return exactly one JSON object with a code field containing the final Python 3 solution."
BASE_GENERATOR_SYSTEM = "Propose a concise algorithm or code improvement for the requested sub-goal."


def query_prompt(question: str, starter_code: str) -> str:
    return f"Problem:\n{question}\n\nStarter code:\n{starter_code or '(empty)'}"


def planner_prompt(question: str, memory_view: str) -> str:
    return f"Problem:\n{question}\n\nMemory:\n{memory_view}\n\nChoose one coding tool."


def executor_prompt(action: str, memory_view: str) -> str:
    return f"Proposed action:\n{action}\n\nMemory:\n{memory_view}"


def verifier_prompt(question: str, memory_view: str) -> str:
    return f"Problem:\n{question}\n\nCode and test memory:\n{memory_view}"


def generator_prompt(question: str, memory_view: str, current_code: str) -> str:
    return f"Problem:\n{question}\n\nMemory:\n{memory_view}\n\nCurrent code:\n{current_code}"
