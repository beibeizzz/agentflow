from typing import Any

from pydantic import BaseModel, ConfigDict

# Planner: QueryAnalysis
class QueryAnalysis(BaseModel):
    concise_summary: str
    required_skills: str
    relevant_tools: str
    additional_considerations: str

    def __str__(self):
        return f"""
Concise Summary: {self.concise_summary}

Required Skills:
{self.required_skills}

Relevant Tools:
{self.relevant_tools}

Additional Considerations:
{self.additional_considerations}
"""

# Planner: NextStep
class NextStep(BaseModel):
    justification: str = ""
    context: str = ""
    calculation: str = ""
    sub_goal: str = ""
    Calculation: str = ""
    Sub_goal: str = ""
    tool_name: str = ""


class StructuredToolAction(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    model_config = ConfigDict(extra="forbid")

# Executor: MemoryVerification
class MemoryVerification(BaseModel):
    analysis: str
    stop_signal: bool

# Executor: ToolCommand
class ToolCommand(BaseModel):
    analysis: str
    explanation: str
    command: str
