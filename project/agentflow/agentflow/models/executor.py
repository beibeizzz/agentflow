import importlib
import json
import os
import re
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from agentflow.engine.factory import create_llm_engine
from agentflow.models.formatters import ToolCommand

# Tool name mapping: Static fallback mapping (long external names to internal)
TOOL_NAME_MAPPING_LONG = {
    "Generalist_Solution_Generator_Tool": {
        "class_name": "Base_Generator_Tool",
        "dir_name": "base_generator"
    },
    "Ground_Google_Search_Tool": {
        "class_name": "Google_Search_Tool",
        "dir_name": "google_search"
    },
    "Python_Code_Generator_Tool": {
        "class_name": "Python_Coder_Tool",
        "dir_name": "python_coder"
    },
    "Web_RAG_Search_Tool": {
        "class_name": "Web_Search_Tool",
        "dir_name": "web_search"
    },
    "Wikipedia_RAG_Search_Tool": {
        "class_name": "Wikipedia_Search_Tool",
        "dir_name": "wikipedia_search"
    }
}

# Short to long mapping for fallback
TOOL_NAME_MAPPING_SHORT = {
    "Base_Generator_Tool": "Generalist_Solution_Generator_Tool",
    "Google_Search_Tool": "Ground_Google_Search_Tool",
    "Python_Coder_Tool": "Python_Code_Generator_Tool",
    "Web_Search_Tool": "Web_RAG_Search_Tool",
    "Wikipedia_Search_Tool": "Wikipedia_RAG_Search_Tool"
}

try:
    TimeoutError
except NameError:
    class TimeoutError(Exception):
        pass


class Executor:
    def __init__(self, llm_engine_name: str, root_cache_dir: str = "solver_cache",  num_threads: int = 1, max_time: int = 120,
    max_output_length: int = 100000, verbose: bool = False, base_url: str = None, check_model: bool = True, temperature: float = .0,
    tool_instances_cache: dict = None, think_mode: str = "default", execution_mode: str = "legacy"):
        if execution_mode not in {"legacy", "structured"}:
            raise ValueError(f"Unsupported Executor execution_mode: {execution_mode}")
        self.llm_engine_name = llm_engine_name
        self.root_cache_dir = root_cache_dir
        self.num_threads = num_threads
        self.max_time = max_time
        self.max_output_length = max_output_length
        self.verbose = verbose
        self.base_url = base_url
        self.check_model = check_model
        self.temperature  = temperature
        self.think_mode = think_mode
        self.execution_mode = execution_mode

        # Store the tool instances cache
        self.tool_instances_cache = tool_instances_cache if tool_instances_cache is not None else {}
        self.generation_configs = {}

        if execution_mode == "structured":
            self.llm_generate_tool_command = None
        elif base_url is not None:
            self.llm_generate_tool_command = create_llm_engine(model_string=self.llm_engine_name, is_multimodal=False, base_url=self.base_url, temperature = self.temperature, think_mode=think_mode)
        else:
            self.llm_generate_tool_command = create_llm_engine(model_string=self.llm_engine_name, is_multimodal=False, temperature = self.temperature, think_mode=think_mode)
    
    def set_query_cache_dir(self, query_cache_dir):
        if getattr(self, "execution_mode", "legacy") == "structured":
            self.query_cache_dir = query_cache_dir
            return
        if query_cache_dir:
            self.query_cache_dir = query_cache_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.query_cache_dir = os.path.join(self.root_cache_dir, timestamp)
        os.makedirs(self.query_cache_dir, exist_ok=True)

    def _think_directive(self) -> str:
        return "/no_think\n" if getattr(self, "think_mode", "default") == "off" else ""

    def generate_tool_command(self, question: str, image: str, context: str, sub_goal: str, tool_name: str, tool_metadata: Dict[str, Any], step_count: int = 0, json_data: Any = None) -> Any:
        if getattr(self, "execution_mode", "legacy") == "structured":
            tool_command = ToolCommand(
                analysis="Deterministic structured dispatch.",
                explanation="Arguments are passed directly to the registered tool.",
                command=context,
            )
            if json_data is not None:
                json_data[f"tool_commander_{step_count}_prompt"] = None
                json_data[f"tool_commander_{step_count}_system_prompt"] = None
                json_data[f"tool_commander_{step_count}_response"] = str(tool_command)
            return tool_command
        if tool_name == "Calculator_Tool":
            prompt_generate_tool_command = f"""
{self._think_directive()}Your should extract the expression of Context into one executable Calculator_Tool command.

Problem: {question}
Context: {context}
Tool: {tool_name}
Tool description: {tool_metadata}

Rules:
- Extract and copy the original arithmetic expression from Context. 
- Do not calculate the result.
- Do not add words, "=" or units inside expression.
- Return only one Python code block and no prose and follow the format below strictly.
- Do not add numbers from the problem.
- Do not combine Context with other calculations.
- Do not change the operation order.

Output Format:

```python
execution = tool.execute(expression="<raw arithmetic expression>")
```

"""
        else:
            prompt_generate_tool_command = f"""
        Task: Generate a precise command to execute the selected tool.

Context:
- **Query:** {question}
- **Sub-Goal:** {sub_goal}
- **Tool Name:** {tool_name}
- **Tool Metadata:** {tool_metadata}
- **Relevant Data:** {context}

Instructions:
1.  Analyze the tool's required parameters from its metadata.
2.  Construct valid Python code that addresses the sub-goal using the provided context and data.
3.  The command must include at least one call to `tool.execute()`.
4.  Each `tool.execute()` call must be assigned to a variable named **`execution`**.
5.  Please give the exact numbers and parameters should be used in the `tool.execute()` call.

Output Format:
Present your response in the following structured format. Do not include any extra text or explanations.

Generated Command:
```python
<command>
```

Example1:
Generated Command:
```python
execution = tool.execute(query="Summarize the following porblom:"Isaac has 100 toys, masa gets ...., how much are their together?")
```

Example2:
Generated Command:
```python
execution = tool.execute(query=["Methanol", "function of hyperbola", "Fermat's Last Theorem"])
```
"""

        generation_kwargs = {"response_format": ToolCommand}
        if tool_name == "Calculator_Tool":
            generation_kwargs.update(getattr(self, "generation_configs", {}).get("executor", {}))
        tool_command = self.llm_generate_tool_command(prompt_generate_tool_command, **generation_kwargs)
        if json_data is not None:
            json_data[f"tool_commander_{step_count}_prompt"] = prompt_generate_tool_command
            json_data[f"tool_commander_{step_count}_system_prompt"] = generation_kwargs.get("system_prompt")
            json_data[f"tool_commander_{step_count}_response"] = str(tool_command)

        return tool_command

    def extract_explanation_and_command(self, response: Any) -> tuple:
        def normalize_code(code: str) -> str:
            # Remove leading/trailing whitespace and triple backticks if present
            return re.sub(r'^```python\s*', '', code).rstrip('```').strip()

        analysis = "No analysis found."
        explanation = "No explanation found."
        command = "No command found."

        if isinstance(response, str):
            text = response.strip()
            # Clean <think>...</think> block if present
            if text.startswith("<think>"):
                think_end = text.find("</think>")
                if think_end != -1:
                    text = text[think_end + len("</think>"):].strip()
            
            # Attempt to parse as JSON first
            try:
                # Need to also clean potential markdown around json block for robust parsing
                clean_text = re.sub(r"^```(?:json)?\s*", "", text)
                clean_text = re.sub(r"\s*```$", "", clean_text)
                response_dict = json.loads(clean_text)
                response_obj = ToolCommand(**response_dict)
                analysis = response_obj.analysis.strip()
                explanation = response_obj.explanation.strip()
                command = response_obj.command.strip()
            except Exception as e:
                print(f"Failed to parse response as JSON: {str(e)}")
                # Fall back to regex parsing on string
                try:
                    # Extract analysis
                    analysis_pattern = r"Analysis:(.*?)Command Explanation"
                    analysis_match = re.search(analysis_pattern, response, re.DOTALL | re.IGNORECASE)
                    analysis = analysis_match.group(1).strip() if analysis_match else "No analysis found."

                    # Extract explanation
                    explanation_pattern = r"Command Explanation:(.*?)Generated Command"
                    explanation_match = re.search(explanation_pattern, response, re.DOTALL | re.IGNORECASE)
                    explanation = explanation_match.group(1).strip() if explanation_match else "No explanation found."

                    # Extract command using "Generated Command:" prefix
                    command_pattern = r"Generated Command:.*?```python\n(.*?)```"
                    command_match = re.search(command_pattern, response, re.DOTALL | re.IGNORECASE)
                    if command_match:
                        command = command_match.group(1).strip()
                    else:
                        # Fallback: Extract ANY ```python ... ``` block (even without prefix)
                        loose_command_pattern = r"```python\s*\n(.*?)```"
                        loose_match = re.findall(loose_command_pattern, response, re.DOTALL | re.IGNORECASE)
                        if loose_match:
                            # Take the last or most complete one? Or first meaningful?
                            # Here we take the longest one as heuristic
                            command = max(loose_match, key=lambda x: len(x.strip())).strip()
                        else:
                            command = "No command found."
                except Exception as e:
                    print(f"Error during regex parsing: {str(e)}")
                    analysis = "Parsing error."
                    explanation = "Parsing error."
                    command = "No command found."
        elif isinstance(response, ToolCommand):
            analysis = response.analysis.strip()
            explanation = response.explanation.strip()
            command = response.command.strip()
        else:
            command = "Invalid response type."

        # Final normalization
        command = normalize_code(command)

        return analysis, explanation, command

    def execute_tool_command(self, tool_name: str, command: str) -> Any:
        """
        Execute a tool command with timeout protection. If execution exceeds max_time seconds,
        the function will be interrupted and return a timeout message.

        Args:
            tool_name (str): Name of the tool to execute
            command (str): Command string containing tool.execute() calls

        Returns:
            Any: List of execution results or error message
        """

        if getattr(self, "execution_mode", "legacy") == "structured":
            try:
                arguments = json.loads(command)
            except (TypeError, json.JSONDecodeError) as exc:
                return {
                    "ok": False,
                    "code": "INVALID_TOOL_ARGUMENTS",
                    "message": str(exc),
                }
            if not isinstance(arguments, dict):
                return {
                    "ok": False,
                    "code": "INVALID_TOOL_ARGUMENTS",
                    "message": "Structured tool arguments must be a JSON object.",
                }
            tool = self.tool_instances_cache.get(tool_name)
            if tool is None:
                return {
                    "ok": False,
                    "code": "UNKNOWN_TOOL",
                    "message": f"Tool {tool_name!r} is not registered.",
                }
            try:
                return tool.execute(**arguments)
            except Exception as exc:
                return {
                    "ok": False,
                    "code": "TOOL_EXECUTION_ERROR",
                    "message": str(exc),
                }

        def split_commands(command: str) -> List[str]:
            # Use regex to find all tool.execute() commands and their surrounding code
            pattern = r'.*?execution\s*=\s*tool\.execute\([^\n]*\)\s*(?:\n|$)'
            blocks = re.findall(pattern, command, re.DOTALL)
            return [block.strip() for block in blocks if block.strip()]

        def execute_with_timeout(block: str, local_context: dict) -> Optional[str]:
            """
            Execute a code block with timeout protection using threading.
            This works in any thread, unlike signal.alarm() which only works in the main thread.
            
            Uses a cancellation event to allow cooperative cancellation and reduce memory leaks.
            """
            import threading
            
            result_container = {'result': None, 'exception': None, 'completed': False}
            cancel_event = threading.Event()
            
            def target():
                try:
                    # Inject cancel_event into the execution context for cooperative cancellation
                    local_context['_cancel_event'] = cancel_event
                    exec(block, globals(), local_context)
                    result_container['result'] = local_context.get('execution')
                    result_container['completed'] = True
                except Exception as e:
                    result_container['exception'] = e
                    result_container['completed'] = True
            
            # Start execution in a daemon thread
            exec_thread = threading.Thread(target=target, name=f"ToolExec-{id(block)}")
            exec_thread.daemon = True
            exec_thread.start()
            
            # Wait for completion or timeout
            exec_thread.join(timeout=self.max_time)
            
            if not result_container['completed']:
                # Timeout occurred - signal cancellation
                cancel_event.set()
                
                # Give it a brief moment to notice the cancellation
                exec_thread.join(timeout=0.5)
                
                # Clean up references to help GC
                result_container.clear()
                local_context.pop('_cancel_event', None)
                
                return f"Execution timed out after {self.max_time} seconds"
            elif result_container['exception']:
                raise result_container['exception']
            else:
                return result_container['result']


        # Try to get tool from cache first
        tool = None

        # Check if tool is in cache (tool_name could be the external long name)
        if tool_name in self.tool_instances_cache:
            tool = self.tool_instances_cache[tool_name]
            print(f"Using cached tool instance for: {tool_name}")
        else:
            # Fallback: Import the tool module and instantiate it
            print(f"Warning: Tool '{tool_name}' not found in cache, instantiating with default parameters")

            # tool_name could be either short or long name
            # First check if it's a long name
            if tool_name in TOOL_NAME_MAPPING_LONG:
                dir_name = TOOL_NAME_MAPPING_LONG[tool_name]["dir_name"]
                class_name = TOOL_NAME_MAPPING_LONG[tool_name]["class_name"]
            # Then check if it's a short name (convert to long, then get internal)
            elif tool_name in TOOL_NAME_MAPPING_SHORT:
                long_name = TOOL_NAME_MAPPING_SHORT[tool_name]
                if long_name in TOOL_NAME_MAPPING_LONG:
                    dir_name = TOOL_NAME_MAPPING_LONG[long_name]["dir_name"]
                    class_name = TOOL_NAME_MAPPING_LONG[long_name]["class_name"]
                else:
                    # Shouldn't happen, but fallback
                    dir_name = tool_name.lower().replace('_tool', '')
                    class_name = tool_name
            else:
                # Fallback to original behavior for unmapped tools
                dir_name = tool_name.lower().replace('_tool', '')
                class_name = tool_name

            module_name = f"tools.{dir_name}.tool"

            try:
                # Dynamically import the module
                module = importlib.import_module(module_name)

                # Get the tool class
                tool_class = getattr(module, class_name)

                tool = tool_class()

            except Exception as e:
                return f"Error importing tool '{tool_name}': {str(e)}"

        if tool is None:
            return f"Error: Could not get tool instance for '{tool_name}'"

        try:
            # Set the custom output directory
            tool.set_custom_output_dir(self.query_cache_dir)

            # Split the command into blocks, execute each one and store execution results
            command_blocks = split_commands(command)
            executions = []

            for block in command_blocks:
                # Create a local context to safely execute the block
                local_context = {'tool': tool}

                # Execute the block with timeout protection
                result = execute_with_timeout(block, local_context)

                if result is not None:
                    executions.append(result)
                else:
                    executions.append(f"No execution captured from block: {block}")

            # Return all the execution results
            return executions
        except Exception as e:
            return f"Error in execute_tool_command: {str(e)}"
