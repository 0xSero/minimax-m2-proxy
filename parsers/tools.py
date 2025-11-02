"""Tool call parser for MiniMax-M2 XML format

Adapted from VLLM's minimax_m2_tool_parser.py
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional


class ToolCall(Dict[str, Any]):
    """OpenAI-format tool call"""

    def __init__(
        self,
        id: str,
        type: str,
        function: Dict[str, str]
    ):
        super().__init__(
            id=id,
            type=type,
            function=function
        )


class ToolCallParser:
    """Parse <minimax:tool_call> XML format to OpenAI tool calls"""

    def __init__(self):
        # Sentinel tokens
        self.tool_call_start_token = "<minimax:tool_call>"
        self.tool_call_end_token = "</minimax:tool_call>"
        self.invoke_start_prefix = "<invoke name="
        self.invoke_end_token = "</invoke>"
        self.parameter_prefix = "<parameter name="
        self.parameter_end_token = "</parameter>"

        # Regex patterns for complete parsing
        self.tool_call_complete_regex = re.compile(
            r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL
        )
        self.invoke_complete_regex = re.compile(
            r"<invoke name=(.*?)</invoke>", re.DOTALL
        )
        self.parameter_complete_regex = re.compile(
            r"<parameter name=(.*?)</parameter>", re.DOTALL
        )

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID"""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _extract_name(self, name_str: str) -> str:
        """Extract name from quoted string"""
        name_str = name_str.strip()
        if (
            name_str.startswith('"') and name_str.endswith('"')
            or name_str.startswith("'") and name_str.endswith("'")
        ):
            return name_str[1:-1]
        return name_str

    def _convert_param_value(self, value: str, param_type: str = "string") -> Any:
        """Convert parameter value to the correct type"""
        if value.lower() == "null":
            return None

        param_type = param_type.lower()
        if param_type in ["string", "str", "text"]:
            return value
        elif param_type in ["integer", "int"]:
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        elif param_type in ["number", "float"]:
            try:
                val = float(value)
                return val if val != int(val) else int(val)
            except (ValueError, TypeError):
                return value
        elif param_type in ["boolean", "bool"]:
            return value.lower() in ["true", "1"]
        elif param_type in ["object", "array"]:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        else:
            # Try JSON parse first, fallback to string
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

    def _parse_single_invoke(
        self,
        invoke_str: str,
        tools: Optional[List[Dict]] = None
    ) -> Optional[Dict[str, Any]]:
        """Parse a single <invoke> block"""
        # Extract function name
        name_match = re.search(r"^([^>]+)", invoke_str)
        if not name_match:
            return None

        function_name = self._extract_name(name_match.group(1))

        # Get parameter configuration
        param_config = {}
        if tools:
            for tool in tools:
                if isinstance(tool, dict):
                    # OpenAI format: {type: "function", function: {name, parameters}}
                    func = tool.get("function", {})
                    if func.get("name") == function_name:
                        params = func.get("parameters", {})
                        if isinstance(params, dict) and "properties" in params:
                            param_config = params["properties"]
                        break

        # Extract parameters
        param_dict = {}
        for match in self.parameter_complete_regex.findall(invoke_str):
            param_match = re.search(r"^([^>]+)>(.*)", match, re.DOTALL)
            if param_match:
                param_name = self._extract_name(param_match.group(1))
                param_value = param_match.group(2).strip()
                if param_value.startswith("\n"):
                    param_value = param_value[1:]
                if param_value.endswith("\n"):
                    param_value = param_value[:-1]

                # Get parameter type
                param_type = "string"
                if (
                    param_name in param_config
                    and isinstance(param_config[param_name], dict)
                    and "type" in param_config[param_name]
                ):
                    param_type = param_config[param_name]["type"]

                # Convert value
                param_dict[param_name] = self._convert_param_value(
                    param_value, param_type
                )

        return {
            "id": self._generate_tool_call_id(),
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": json.dumps(param_dict, ensure_ascii=False)
            }
        }

    def parse_tool_calls(
        self,
        text: str,
        tools: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Extract tool calls from complete text (non-streaming).

        Returns:
            {
                "tools_called": bool,
                "tool_calls": List[Dict],
                "content": str  # Content without tool call blocks
            }
        """
        # Quick check
        if self.tool_call_start_token not in text:
            return {
                "tools_called": False,
                "tool_calls": [],
                "content": text
            }

        try:
            tool_calls = []

            # Find all complete tool_call blocks
            for tool_call_match in self.tool_call_complete_regex.findall(text):
                # Find all invokes within this tool_call
                for invoke_match in self.invoke_complete_regex.findall(tool_call_match):
                    tool_call = self._parse_single_invoke(invoke_match, tools)
                    if tool_call:
                        tool_calls.append(tool_call)

            if not tool_calls:
                return {
                    "tools_called": False,
                    "tool_calls": [],
                    "content": text
                }

            # Extract content before first tool call
            first_tool_idx = text.find(self.tool_call_start_token)
            content = text[:first_tool_idx].strip() if first_tool_idx > 0 else None

            return {
                "tools_called": True,
                "tool_calls": tool_calls,
                "content": content
            }

        except Exception as e:
            print(f"Error extracting tool calls: {e}")
            return {
                "tools_called": False,
                "tool_calls": [],
                "content": text
            }

    def extract_content_without_tools(self, text: str) -> str:
        """Remove tool call blocks, keep only text"""
        return self.tool_call_complete_regex.sub('', text).strip()

    def has_tool_calls(self, text: str) -> bool:
        """Check if text contains tool calls"""
        return self.tool_call_start_token in text
