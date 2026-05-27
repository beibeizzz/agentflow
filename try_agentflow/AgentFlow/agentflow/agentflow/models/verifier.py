import json
import os
import re
from typing import Any, Tuple

from PIL import Image

from agentflow.engine.factory import create_llm_engine
from agentflow.models.formatters import MemoryVerification
from agentflow.models.memory import Memory


class Verifier:
    def __init__(self, llm_engine_name: str, llm_engine_fixed_name: str = "dashscope",
                 toolbox_metadata: dict = None, available_tools: list = None,
                 verbose: bool = False, base_url: str = None, is_multimodal: bool = False,
                 check_model: bool = True, temperature: float = .0):
        self.llm_engine_name = llm_engine_name
        self.llm_engine_fixed_name = llm_engine_fixed_name
        self.is_multimodal = is_multimodal
        self.llm_engine_fixed = create_llm_engine(model_string=llm_engine_fixed_name, is_multimodal=False, temperature=temperature)
        self.llm_engine = create_llm_engine(model_string=llm_engine_name, is_multimodal=False, base_url=base_url, temperature=temperature)
        self.toolbox_metadata = toolbox_metadata if toolbox_metadata is not None else {}
        self.available_tools = available_tools if available_tools is not None else []
        self.verbose = verbose
        self.generation_configs = {}

    def get_image_info(self, image_path: str) -> dict:
        image_info = {}
        if image_path and os.path.isfile(image_path):
            image_info["image_path"] = image_path
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                image_info.update({
                    "width": width,
                    "height": height
                })
            except Exception as e:
                print(f"Error processing image file: {str(e)}")
        return image_info

    def verificate_context(self, question: str, image: str, query_analysis: str, memory: Memory, step_count: int = 0, json_data: Any = None) -> Any:
        image_info = self.get_image_info(image)
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

        if self.is_multimodal:
            prompt_memory_verification = f"""
Task: Thoroughly evaluate the completeness and accuracy of the memory for fulfilling the given query, considering the potential need for additional tool usage.

Context:
Query: {question}
Image: {image_info}
Available Tools: {self.available_tools}
Toolbox Metadata: {self.toolbox_metadata}
Initial Analysis: {query_analysis}
Memory (tools used and results): {memory.get_actions()}

Detailed Instructions:
1. Carefully analyze the query, initial analysis, and image (if provided):
   - Identify the main objectives of the query.
   - Note any specific requirements or constraints mentioned.
   - If an image is provided, consider its relevance and what information it contributes.

2. Review the available tools and their metadata:
   - Understand the capabilities and limitations and best practices of each tool.
   - Consider how each tool might be applicable to the query.

3. Examine the memory content in detail:
   - Review each tool used and its execution results.
   - Assess how well each tool's output contributes to answering the query.

4. Critical Evaluation (address each point explicitly):
   a) Completeness: Does the memory fully address all aspects of the query?
      - Identify any parts of the query that remain unanswered.
      - Consider if all relevant information has been extracted from the image (if applicable).

   b) Unused Tools: Are there any unused tools that could provide additional relevant information?
      - Specify which unused tools might be helpful and why.

   c) Inconsistencies: Are there any contradictions or conflicts in the information provided?
      - If yes, explain the inconsistencies and suggest how they might be resolved.

   d) Verification Needs: Is there any information that requires further verification due to tool limitations?
      - Identify specific pieces of information that need verification and explain why.

   e) Ambiguities: Are there any unclear or ambiguous results that could be clarified by using another tool?
      - Point out specific ambiguities and suggest which tools could help clarify them.

5. Final Determination:
   Based on your thorough analysis, decide if the memory is complete and accurate enough to generate the final output, or if additional tool usage is necessary.

Response Format:

If the memory is complete, accurate, AND verified:
Explanation:
<Provide a detailed explanation of why the memory is sufficient. Reference specific information from the memory and explain its relevance to each aspect of the task. Address how each main point of the query has been satisfied.>

Conclusion: STOP

If the memory is incomplete, insufficient, or requires further verification:
Explanation:
<Explain in detail why the memory is incomplete. Identify specific information gaps or unaddressed aspects of the query. Suggest which additional tools could be used, how they might contribute, and why their input is necessary for a comprehensive response.>

Conclusion: CONTINUE

IMPORTANT: Your response MUST end with either 'Conclusion: STOP' or 'Conclusion: CONTINUE' and nothing else. Ensure your explanation thoroughly justifies this conclusion.
"""
        elif calculator_only:
            prompt_memory_verification = f"""
Decide whether memory has enough proof to solve the entire problem.
Inspect Memory step by step.
Initial Analysis and action_predictor_response is for reference only.
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
- Don't solves the problem or repeat the raw problem. 
- Analyse the missing logic if neccessary. 
- Follow the formats above. Response only one of the two formats below.
- Do not write another Conclusion later.
- First line must be Conclusion. Do not write any other Conclusion in the response. 

Response Format:
Format1 (When memory not solves the entire problem):
Conclusion: CONTINUE
Current memory can't solve the problem. 
Current issue: ... 
Next action:...
<end>

Format2 (Only when memory solves the entire problem):
Conclusion: STOP
Current memory solve the entire problem.
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
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        generation_kwargs = {"response_format": MemoryVerification}
        if calculator_only:
            generation_kwargs.update(getattr(self, "generation_configs", {}).get("verifier", {}))
        stop_verification = self.llm_engine_fixed(input_data, **generation_kwargs)
        if json_data is not None:
            json_data[f"verifier_{step_count}_prompt"] = input_data
            json_data[f"verifier_{step_count}_system_prompt"] = generation_kwargs.get("system_prompt")
            json_data[f"verifier_{step_count}_generation_config"] = {
                key: value
                for key, value in generation_kwargs.items()
                if key not in {"response_format", "system_prompt"}
            }
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
