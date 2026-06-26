import json
import re
from typing import Any, Tuple


from agentflow.engine.factory import create_llm_engine
from agentflow.models.formatters import MemoryVerification
from agentflow.models.memory import Memory


class Verifier:
    def __init__(self, llm_engine_name: str, llm_engine_fixed_name: str = "dashscope",
                 toolbox_metadata: dict = None, available_tools: list = None,
                 verbose: bool = False, base_url: str = None, is_multimodal: bool = False,
                 check_model: bool = True, temperature: float = .0,
                 think_mode: str = "default",
                 verifier_think_mode: str = None):
        self.llm_engine_name = llm_engine_name
        self.llm_engine_fixed_name = llm_engine_fixed_name
        self.think_mode = think_mode
        self.verifier_think_mode = verifier_think_mode or think_mode
        self.llm_engine_fixed = create_llm_engine(model_string=llm_engine_fixed_name, is_multimodal=False, temperature=temperature, think_mode=think_mode)
        self.llm_engine = create_llm_engine(model_string=llm_engine_name, is_multimodal=False, base_url=base_url, temperature=temperature, think_mode=think_mode)
        self.toolbox_metadata = toolbox_metadata if toolbox_metadata is not None else {}
        self.available_tools = available_tools if available_tools is not None else []
        self.verbose = verbose
        self.generation_configs = {}

    def _think_directive(self) -> str:
        think_mode = getattr(self, "verifier_think_mode", getattr(self, "think_mode", "default"))
        return "/no_think\n" if think_mode == "off" else ""

    def verificate_context(self, question: str, image: str, query_analysis: str, memory: Memory, step_count: int = 0, json_data: Any = None) -> Any:
        calculator_only = self.available_tools == ["Calculator_Tool"]
        memory_actions = memory.get_actions()
        if calculator_only and memory_actions:
            last_step_key = sorted(
                memory_actions,
                key=lambda name: int(name.rsplit(" ", 1)[-1]) if name.rsplit(" ", 1)[-1].isdigit() else -1,
            )[-1]
            last_action = memory_actions.get(last_step_key, {})
            if last_action.get("command") == "Planner output was invalid; no tool command generated.":
                fixed_response = "Conclusion: CONTINUE\nlast Planner output was invalid. check the Calculation first"
                if json_data is not None:
                    json_data[f"verifier_{step_count}_response"] = fixed_response
                return fixed_response

        if calculator_only:
            prompt_memory_verification = f"""
{self._think_directive()}Decide whether memory has enough proof to solve the entire problem.
Initial Analysis and Memory's action_predictor_response is only a hint, not proof.
Command/result pairs from executed tools count as proof.

Before STOP, check:
- What exact quantity does the problem ask for?
- What exact quantity did the latest command compute?
- Are they the same quantity?
If you are not sure about confirming the above questions, output Conclusion: CONTINUE.

Context:
- Problem: {question}
- Initial Analysis: {query_analysis}
- Memory: {memory_actions}

Rules:
- First line must be Conclusion. Do not write any other Conclusion in the response. 
- Do not solve the problem or repeat the raw problem. 
- Analyse the missing logic if neccessary. 
- Follow the formats above. Response only one of the two formats below.
- Do not write another Conclusion later.


Response Format:
Format1 (When memory not solves the entire problem):
Conclusion: CONTINUE
Current memory can't solve the problem. 
Current issue: ... 
Next action:...
<end>

Format2 (Only when memory solves the entire problem):
Conclusion: STOP
Current memory solves the entire problem.
<end>





"""
        else:
            prompt_memory_verification = f"""
Task: Evaluate if the current memory is complete and accurate enough to answer the query, or if more tools are needed.

Context:
- **Query:** {question}
- **Available Tools:** {self.available_tools}
- **Toolbox Metadata:** {self.toolbox_metadata}
- **Initial Analysis:** {query_analysis}
- **Memory (Tools Used & Results):** {memory.get_actions()}

Instructions:
1.  Review the query, initial analysis, and memory.
2.  Assess the completeness of the memory: Does it fully address all parts of the query?
3.  Check for potential issues:
    -   Are there any inconsistencies or contradictions?
    -   Is any information ambiguous or in need of verification?
4.  Determine if any unused tools could provide missing information.

Final Determination:
-   If the memory is sufficient, explain why and conclude with "STOP".
-   If more information is needed, explain what's missing, which tools could help, and conclude with "CONTINUE".

IMPORTANT: The response must end with either "Conclusion: STOP" or "Conclusion: CONTINUE".
"""

        input_data = [prompt_memory_verification]
        generation_kwargs = {"response_format": MemoryVerification}
        if calculator_only:
            generation_kwargs.update(getattr(self, "generation_configs", {}).get("verifier", {}))
        verifier_think_mode = getattr(self, "verifier_think_mode", getattr(self, "think_mode", "default"))
        if verifier_think_mode != "default":
            generation_kwargs["think_mode"] = verifier_think_mode
        stop_verification = self.llm_engine_fixed(input_data, **generation_kwargs)
        if json_data is not None:
            json_data[f"verifier_{step_count}_prompt"] = input_data
            json_data[f"verifier_{step_count}_system_prompt"] = generation_kwargs.get("system_prompt")
            json_data[f"verifier_{step_count}_response"] = str(stop_verification)
        return stop_verification

    def extract_conclusion(self, response: Any) -> Tuple[str, str]:
        if isinstance(response, str):
            # Clean <think>...</think> block if present
            stripped_response = response.strip()
            if stripped_response.startswith("<think>"):
                think_end = stripped_response.find("</think>")
                if think_end != -1:
                    stripped_response = stripped_response[think_end + len("</think>"):].strip()
            
            # Attempt to parse the response as JSON
            if stripped_response.startswith("{"):
                try:
                    response_dict = json.loads(stripped_response)
                    response = MemoryVerification(**response_dict)
                except Exception as e:
                    print(f"Failed to parse response as JSON: {str(e)}")
            elif stripped_response.startswith("```"):
                # Clean markdown fences too
                clean_text = re.sub(r"^```(?:json)?\s*", "", stripped_response)
                clean_text = re.sub(r"\s*```$", "", clean_text)
                if clean_text.startswith("{"):
                    try:
                        response_dict = json.loads(clean_text)
                        response = MemoryVerification(**response_dict)
                    except Exception as e:
                        pass
            
            # if we didn't transform response to MemoryVerification, keep stripped_response for fallback
            if isinstance(response, str):
                response = stripped_response

        if isinstance(response, MemoryVerification):
            analysis = response.analysis
            stop_signal = response.stop_signal
            if stop_signal:
                return analysis, 'STOP'
            else:
                return analysis, 'CONTINUE'
        else:
            analysis = response
            calculator_only = getattr(self, "available_tools", []) == ["Calculator_Tool"]
            if calculator_only:
                first_line = next(
                    (line.strip() for line in response.splitlines() if line.strip()),
                    "",
                )
                match = re.fullmatch(r"Conclusion:\s*(STOP|CONTINUE)", first_line, re.IGNORECASE)
                if match:
                    return analysis, match.group(1).upper()
                print("No valid first-line conclusion (STOP or CONTINUE) found in the response. Continuing...")
                return analysis, 'CONTINUE'

            pattern = r'conclusion\**:?\s*\**\s*(\w+)'
            matches = list(re.finditer(pattern, response, re.IGNORECASE | re.DOTALL))
            if matches:
                conclusion = matches[-1].group(1).upper()
                if conclusion in ['STOP', 'CONTINUE']:
                    return analysis, conclusion

            # If no valid conclusion found, search for STOP or CONTINUE anywhere in the text
            if 'stop' in response.lower():
                return analysis, 'STOP'
            elif 'continue' in response.lower():
                return analysis, 'CONTINUE'
            else:
                print("No valid conclusion (STOP or CONTINUE) found in the response. Continuing...")
                return analysis, 'CONTINUE'
