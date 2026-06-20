import json
import os
import re
from typing import Any, Dict, List, Tuple

from PIL import Image

from agentflow.engine.factory import create_llm_engine
from agentflow.models.formatters import NextStep, QueryAnalysis
from agentflow.models.memory import Memory


DEFAULT_GSM8K_DIRECT_SYSTEM_PROMPT = (
    "You are good at math problems. "
    "Use only the information in the problem. "
    "Keep the reasoning concise and arithmetic-focused."
)


class Planner:
    def __init__(self, llm_engine_name: str, llm_engine_fixed_name: str = "gpt-4o",
                 toolbox_metadata: dict = None, available_tools: List = None,
                 verbose: bool = False, base_url: str = None, is_multimodal: bool = False,
                 check_model: bool = True, temperature : float = .0,
                 think_mode: str = "default",
                 query_analysis_think_mode: str = None,
                 final_output_think_mode: str = None):
        self.llm_engine_name = llm_engine_name
        self.llm_engine_fixed_name = llm_engine_fixed_name
        self.is_multimodal = is_multimodal
        self.think_mode = think_mode
        self.query_analysis_think_mode = query_analysis_think_mode or think_mode
        self.final_output_think_mode = final_output_think_mode or think_mode
        # self.llm_engine_mm = create_llm_engine(model_string=llm_engine_name, is_multimodal=False, base_url=base_url, temperature = temperature)
        self.llm_engine_fixed = create_llm_engine(model_string=llm_engine_fixed_name, is_multimodal=False, temperature = temperature, think_mode=think_mode)
        self.llm_engine = create_llm_engine(model_string=llm_engine_name, is_multimodal=False, base_url=base_url, temperature = temperature, think_mode=think_mode)
        self.toolbox_metadata = toolbox_metadata if toolbox_metadata is not None else {}
        self.available_tools = available_tools if available_tools is not None else []
        self.generation_configs = {}

        self.verbose = verbose

    def _generation_config(self, key: str) -> Dict[str, Any]:
        return dict(getattr(self, "generation_configs", {}).get(key, {}))

    def _think_directive(self) -> str:
        return "/no_think\n" if getattr(self, "think_mode", "default") == "off" else ""

    def get_image_info(self, image_path: str) -> Dict[str, Any]:
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

    def generate_base_response(self, question: str, image: str, max_tokens: int = 2048) -> str:
        image_info = self.get_image_info(image)

        input_data = [question]
        if image_info and "image_path" in image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")


        print("Input data of `generate_base_response()`: ", input_data)
        self.base_response = self.llm_engine(input_data, max_tokens=max_tokens)
        # self.base_response = self.llm_engine_fixed(input_data, max_tokens=max_tokens)

        return self.base_response

    def analyze_query(self, question: str, image: str, json_data: Any = None) -> str:
        image_info = self.get_image_info(image)
        calculator_only = self.available_tools == ["Calculator_Tool"]
        query_config = self._generation_config("query_analysis") if calculator_only else {}

        if calculator_only and query_config.get("enabled") is False:
            self.query_analysis = ""
            if json_data is not None:
                json_data["query_analysis_disabled"] = True
                json_data["query_analysis_prompt"] = None
                json_data["query_analysis_system_prompt"] = query_config.get("system_prompt")
            return ""

        if self.is_multimodal:
            query_prompt = f"""
Task: Analyze the given query with accompanying inputs and determine the skills and tools needed to address it effectively.

Available tools: {self.available_tools}

Metadata for the tools: {self.toolbox_metadata}

Image: {image_info}

Query: {question}

Instructions:
1. Carefully read and understand the query and any accompanying inputs.
2. Identify the main objectives or tasks within the query.
3. List the specific skills that would be necessary to address the query comprehensively.
4. Examine the available tools in the toolbox and determine which ones might relevant and useful for addressing the query. Make sure to consider the user metadata for each tool, including limitations and potential applications (if available).
5. Provide a brief explanation for each skill and tool you've identified, describing how it would contribute to answering the query.

Your response should include:
1. A concise summary of the query's main points and objectives, as well as content in any accompanying inputs.
2. A list of required skills, with a brief explanation for each.
3. A list of relevant tools from the toolbox, with a brief explanation of how each tool would be utilized and its potential limitations.
4. Any additional considerations that might be important for addressing the query effectively.

Please present your analysis in a clear, structured format.
                        """
        elif calculator_only:
            query_prompt = f"""
Explain the general solution approach step by step and the final goal to the problem in a concise manner, without delving into specific calculations.

Inputs:
- Problem: {question}

Rules:
- Do not calculate numerically.
- Do not give the final answer or generator number.
- Do not determine the specific values of the intermediate variables or the final target.

"""

#             f"""Solve the following GSM8K math word problem.

# Instructions:
# - Reason step by step before the final answer.
# - Use only facts stated in the problem.
# - Do not introduce extra days, weeks, people, prices, or assumptions.
# - The final answer must be a single numeric value, integer, decimal, or fraction.
# - Do not include units, currency symbols, commas, or explanatory text inside the answer tag.
# - End with exactly one final line in this format:
# <answer>NUMBER</answer>
# - No text after </answer>.

# Problem:
# {question}"""

        else:
            query_prompt = f"""Solve the following GSM8K math word problem.

Instructions:
- Reason step by step before the final answer.
- Use only facts stated in the problem.
- Do not introduce extra days, weeks, people, prices, or assumptions.
- The final answer must be a single numeric value, integer, decimal, or fraction.
- Do not include units, currency symbols, commas, or explanatory text inside the answer tag.
- End with exactly one final line in this format:
<answer>NUMBER</answer>
- No text after </answer>.

Problem:
{question}"""

# f"""
# Task: Analyze the given query to determine necessary skills and tools.

# Inputs:
# - Query: {question}
# - Available tools: {self.available_tools}
# - Metadata for tools: {self.toolbox_metadata}

# Instructions:
# 1. Identify the main objectives in the query.
# 2. List the necessary skills and tools.
# 3. For each skill and tool, explain how it helps address the query.
# 4. Note any additional considerations.

# """


        input_data = [query_prompt]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        print("Input data of `analyze_query()`: ", input_data)

        # self.query_analysis = self.llm_engine_mm(input_data, response_format=QueryAnalysis)
        # self.query_analysis = self.llm_engine(input_data, response_format=QueryAnalysis)
        query_kwargs = {"response_format": QueryAnalysis}
        if calculator_only:
            query_kwargs.update({
                "system_prompt": DEFAULT_GSM8K_DIRECT_SYSTEM_PROMPT,
                "max_tokens": 512,
                "temperature": 0.0,
                "top_p": 0.95,
                "frequency_penalty": 0,
            })
            query_kwargs.update(
                {
                    key: value
                    for key, value in query_config.items()
                    if key != "enabled"
                }
            )
        query_think_mode = getattr(self, "query_analysis_think_mode", getattr(self, "think_mode", "default"))
        if query_think_mode != "default":
            query_kwargs["think_mode"] = query_think_mode
        if json_data is not None:
            json_data["query_analysis_prompt"] = input_data
            json_data["query_analysis_system_prompt"] = query_kwargs.get("system_prompt")
        self.query_analysis = self.llm_engine_fixed(input_data, **query_kwargs)

        return str(self.query_analysis).strip()

    def extract_context_subgoal_and_tool(self, response: Any) -> Tuple[str, str, str]:
        calculator_only = self.available_tools == ["Calculator_Tool"]
        calculator_sub_goal = "Calculate the next arithmetic expression"
        calculator_tool_name = "Calculator_Tool"

        def calculator_step(context: str, sub_goal: str) -> Tuple[str, str, str]:
            return (
                context.strip(),
                (sub_goal or calculator_sub_goal).strip(),
                calculator_tool_name,
            )

        def normalize_tool_name(tool_name: str) -> str:
            def to_canonical(name: str) -> str:
                parts = re.split(r"[ _]+", str(name))
                return "_".join(part.lower() for part in parts if part)

            normalized_input = to_canonical(tool_name)
            for tool in self.available_tools:
                if to_canonical(tool) == normalized_input:
                    return tool
            return f"No matched tool given: {tool_name}"

        try:
            # Case 1: already a NextStep object
            if isinstance(response, NextStep):
                context = response.context or response.calculation or response.Calculation
                sub_goal = response.sub_goal or response.Sub_goal
                tool_name = response.tool_name
                if calculator_only and context:
                    return calculator_step(context, sub_goal)
                if context and sub_goal and tool_name:
                    return context.strip(), sub_goal.strip(), normalize_tool_name(tool_name)
                print(f"Could not parse planner response: {response!r}")
                return None, None, None

            # Case 2: JSON string from small model
            if isinstance(response, str):
                text = response.strip()

                # Clean <think>...</think> block if present
                if text.startswith("<think>"):
                    think_end = text.find("</think>")
                    if think_end != -1:
                        text = text[think_end + len("</think>"):].strip()

                # remove markdown fences if any
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)

                try:
                    obj = json.loads(text)
                    if isinstance(obj, dict):
                        context = str(obj.get("context") or obj.get("calculation") or obj.get("Calculation") or "").strip()
                        sub_goal = str(obj.get("sub_goal") or obj.get("Sub_goal") or "").strip()
                        tool_name = str(obj.get("tool_name", "")).strip()

                        if calculator_only and context:
                            return calculator_step(context, sub_goal)
                        if context and sub_goal and tool_name:
                            return context, sub_goal, normalize_tool_name(tool_name)
                except Exception as e:
                    print(f"JSON parse failed in extract_context_subgoal_and_tool: {e}")

                # Case 3: old text format fallback
                plain = text.replace("**", "")
                pattern = r"Context:\s*(.*?)Sub-Goal:\s*(.*?)Tool Name:\s*(.*?)\s*(?:```)?\s*(?=\n\n|\Z)"
                matches = re.findall(pattern, plain, re.DOTALL)
                if matches:
                    context, sub_goal, tool_name = matches[-1]
                    return context.strip(), sub_goal.strip(), normalize_tool_name(tool_name.strip())

            print(f"Could not parse planner response: {response!r}")
            return None, None, None

        except Exception as e:
            print(f"Error extracting context, sub-goal, and tool name: {e}")
            return None, None, None



    def generate_next_step(self, question: str, image: str, query_analysis: str, memory: Memory, step_count: int, max_step_count: int, json_data: Any = None) -> Any:
        calculator_only = self.available_tools == ["Calculator_Tool"]
        if self.is_multimodal:
            prompt_generate_next_step = f"""
Task: Determine the optimal next step to address the given query based on the provided analysis, available tools, and previous steps taken.

Context:
Query: {question}
Image: {image}
Query Analysis: {query_analysis}

Available Tools:
{self.available_tools}

Tool Metadata:
{self.toolbox_metadata}

Previous Steps and Their Results:
{memory.get_actions()}

Current Step: {step_count} in {max_step_count} steps
Remaining Steps: {max_step_count - step_count}

Instructions:
1. Analyze the context thoroughly, including the query, its analysis, any image, available tools and their metadata, and previous steps taken.

2. Determine the most appropriate next step by considering:
- Key objectives from the query analysis
- Capabilities of available tools
- Logical progression of problem-solving
- Outcomes from previous steps
- Current step count and remaining steps

3. Select ONE tool best suited for the next step, keeping in mind the limited number of remaining steps.

4. Formulate a specific, achievable sub-goal for the selected tool that maximizes progress towards answering the query.

Response Format:
Your response MUST follow this structure:
1. Justification: Explain your choice in detail.
2. Context, Sub-Goal, and Tool: Present the context, sub-goal, and the selected tool ONCE with the following format:

Context: <context>
Sub-Goal: <sub_goal>
Tool Name: <tool_name>

Where:
- <context> MUST include ALL necessary information for the tool to function, structured as follows:
* Relevant data from previous steps
* File names or paths created or used in previous steps (list EACH ONE individually)
* Variable names and their values from previous steps' results
* Any other context-specific information required by the tool
- <sub_goal> is a specific, achievable objective for the tool, based on its metadata and previous outcomes.
It MUST contain any involved data, file names, and variables from Previous Steps and Their Results that the tool can act upon.
- <tool_name> MUST be the exact name of a tool from the available tools list.

Rules:
- Select only ONE tool for this step.
- The sub-goal MUST directly address the query and be achievable by the selected tool.
- The Context section MUST include ALL necessary information for the tool to function, including ALL relevant file paths, data, and variables from previous steps.
- The tool name MUST exactly match one from the available tools list: {self.available_tools}.
- Avoid redundancy by considering previous steps and building on prior results.
- Your response MUST conclude with the Context, Sub-Goal, and Tool Name sections IN THIS ORDER, presented ONLY ONCE.
- Include NO content after these three sections.

Example (do not copy, use only as reference):
Justification: [Your detailed explanation here]
Context: Image path: "example/image.jpg", Previous detection results: [list of objects]
Sub-Goal: Detect and count the number of specific objects in the image "example/image.jpg"
Tool Name: Object_Detector_Tool

Remember: Your response MUST end with the Context, Sub-Goal, and Tool Name sections, with NO additional content afterwards.
                        """
        elif calculator_only:
            prompt_generate_next_step = f"""
{self._think_directive()}You should choose the next calculator step and provide the arithmetic expression.
You are strict to output a JSON. 
Use the Judge feedback first.
Do not repeat any previous Calculation or Sub_goal in Memory.
Problem: {question}
Query Analysis: {query_analysis}
Memory: {memory.get_actions()}

Rules:
- Return only one JSON object.
- "Sub_goal": briefly say what this calculation computes.
- "Calculation": write only the arithmetic expression and must match this regex: ^[0-9+\-*/(). ]+$
- In calculation, use only digits, +, -, *, /, parentheses, and decimals.
- In calculation, do not include variables, words, units, "=", , currency symbols, commas, explanatory text, or the result.

JSON example:
{{
  "Sub_goal": "Calculate reading time per night",
  "Calculation": "2 / 2"
}}

Important:
- Replace the placeholder contents with values specific to the current problem.
- Do not copy the example text or expression.

"""
        else:
            prompt_generate_next_step = f"""
Task: Determine the optimal next step to address the query using available tools and previous steps.

Context:
- **Query:** {question}
- **Query Analysis:** {query_analysis}
- **Available Tools:** {self.available_tools}
- **Toolbox Metadata:** {self.toolbox_metadata}
- **Previous Steps:** {memory.get_actions()}

Instructions:
1. Analyze the query, previous steps, and available tools.
2. Select the **single best tool** for the next step.
3. Formulate a specific, achievable **sub-goal** for that tool.
4. Provide all necessary **context** (data, file names, variables) for the tool to function.

Response Format:
1.  **Justification:** Explain your choice of tool and sub-goal.
2.  **Context:** Provide all necessary information for the tool.
3.  **Sub-Goal:** State the specific objective for the tool.
4.  **Tool Name:** State the exact name of the selected tool.

Rules:
- Select only ONE tool.
- The sub-goal must be directly achievable by the selected tool.
- The Context section must contain all information the tool needs to function.
- The response must end with the Context, Sub-Goal, and Tool Name sections in that order, with no extra content.
                    """
            
        generation_kwargs = {"response_format": NextStep}
        if calculator_only:
            generation_kwargs.update(self._generation_config("planner_next_step"))
        next_step = self.llm_engine(prompt_generate_next_step, **generation_kwargs)
        if json_data is not None:
            json_data[f"action_predictor_{step_count}_prompt"] = prompt_generate_next_step
            json_data[f"action_predictor_{step_count}_system_prompt"] = generation_kwargs.get("system_prompt")
            json_data[f"action_predictor_{step_count}_response"] = str(next_step)
        return next_step


    def generate_final_output(self, question: str, image: str, memory: Memory, json_data: Any = None) -> str:
        image_info = self.get_image_info(image)
        calculator_only = self.available_tools == ["Calculator_Tool"]
        if self.is_multimodal:
            prompt_generate_final_output = f"""
Task: Generate the final output based on the query, image, and tools used in the process.

Context:
Query: {question}
Image: {image_info}
Actions Taken:
{memory.get_actions()}

Instructions:
1. Review the query, image, and all actions taken during the process.
2. Consider the results obtained from each tool execution.
3. Incorporate the relevant information from the memory to generate the step-by-step final output.
4. The final output should be consistent and coherent using the results from the tools.

Output Structure:
Your response should be well-organized and include the following sections:

1. Summary:
   - Provide a brief overview of the query and the main findings.

2. Detailed Analysis:
   - Break down the process of answering the query step-by-step.
   - For each step, mention the tool used, its purpose, and the key results obtained.
   - Explain how each step contributed to addressing the query.

3. Key Findings:
   - List the most important discoveries or insights gained from the analysis.
   - Highlight any unexpected or particularly interesting results.

4. Answer to the Query:
   - Directly address the original question with a clear and concise answer.
   - If the query has multiple parts, ensure each part is answered separately.

5. Additional Insights (if applicable):
   - Provide any relevant information or insights that go beyond the direct answer to the query.
   - Discuss any limitations or areas of uncertainty in the analysis.

6. Conclusion:
   - Summarize the main points and reinforce the answer to the query.
   - If appropriate, suggest potential next steps or areas for further investigation.
"""
        elif calculator_only:
                prompt_generate_final_output = f"""
Problem:
{question}

Calculator results:
{memory.get_actions()}

Task:
Return the final numeric answer only.

Rules:
- In each memory action, command contains the arithmetic expression and result contains only the numeric calculator output.
- Calculator results may be unreliable because commands can be incomplete, repeated, or based on a wrong expression.
- Check whether the commands cover every required quantity in the problem.
- If calculator results are complete and consistent, use the final relevant calculator result.
- If calculator results are incomplete or inconsistent, compute the final answer from the problem and memory.
- Do not explain.
- Do not add units.
- Output one number only.
"""
        else:
                prompt_generate_final_output = f"""
Task: Generate the final output based on the query and the results from all tools used.

Context:
- **Query:** {question}
- **Actions Taken:** {memory.get_actions()}

Instructions:
1. Review the query and the results from all tool executions.
2. Incorporate the relevant information to create a coherent, step-by-step final output.
"""

        input_data = [prompt_generate_final_output]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        # final_output = self.llm_engine_mm(input_data)
        # final_output = self.llm_engine(input_data)
        generation_kwargs = self._generation_config("generator") if calculator_only else {}
        final_think_mode = getattr(self, "final_output_think_mode", getattr(self, "think_mode", "default"))
        if final_think_mode != "default":
            generation_kwargs["think_mode"] = final_think_mode
        if json_data is not None:
            json_data["final_output_prompt"] = input_data
            json_data["final_output_system_prompt"] = generation_kwargs.get("system_prompt")
        final_output = self.llm_engine_fixed(input_data, **generation_kwargs)

        return final_output


    def generate_direct_output(self, question: str, image: str, memory: Memory, json_data: Any = None) -> str:
        image_info = self.get_image_info(image)
        calculator_only = self.available_tools == ["Calculator_Tool"]
        if self.is_multimodal:
            prompt_generate_final_output = f"""
Context:
Query: {question}
Image: {image_info}
Initial Analysis:
{self.query_analysis}
Actions Taken:
{memory.get_actions()}

Please generate the concise output based on the query, image information, initial analysis, and actions taken. Break down the process into clear, logical, and conherent steps. Conclude with a precise and direct answer to the query.

Answer:
"""
        elif calculator_only:
            prompt_generate_final_output = f"""
Return the final numeric answer based on the comprehensive Analysis and Memory.

Problem:{question}
Analysis: {self.query_analysis}
Memory:{memory.get_actions()}

Rules:
- Memory contains the previous sub-goals and command actions.
- Memory may be unreliable because commands can be incomplete, repeated, or based on a wrong expression.
- Check whether the commands cover every required quantity to solve the problem.
- If Memory are complete and consistent, refer to the final relevant calculator result.
- If Memory are incomplete or inconsistent, refer to the the problem and Analysis.
- Do not explain.
- Output one number only.
"""
        else:
            prompt_generate_final_output = f"""
Task: Generate a concise final answer to the query based on all provided context.

Context:
- **Query:** {question}
- **Initial Analysis:** {self.query_analysis}
- **Actions Taken:** {memory.get_actions()}

Instructions:
1. Review the query and the results from all actions.
2. Synthesize the key findings into a clear, step-by-step summary of the process.
3. Provide a direct, precise answer to the original query.

Output Structure:
1.  **Process Summary:** A clear, step-by-step breakdown of how the query was addressed, including the purpose and key results of each action.
2.  **Answer:** A direct and concise final answer to the query.
"""

        input_data = [prompt_generate_final_output]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        # final_output = self.llm_engine(input_data)
        generation_kwargs = self._generation_config("generator") if calculator_only else {}
        final_think_mode = getattr(self, "final_output_think_mode", getattr(self, "think_mode", "default"))
        if final_think_mode != "default":
            generation_kwargs["think_mode"] = final_think_mode
        if json_data is not None:
            json_data["direct_output_prompt"] = input_data
            json_data["direct_output_system_prompt"] = generation_kwargs.get("system_prompt")
        final_output = self.llm_engine_fixed(input_data, **generation_kwargs)
        # final_output = self.llm_engine_mm(input_data)

        return final_output
