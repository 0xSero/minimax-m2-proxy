"""Tool call parser for MiniMax-M2 XML format

Based on official MiniMax guide:
https://platform.minimax.io/docs/guides/text-m2-function-call
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional


def _format_param_value_for_xml(value: Any) -> str:
    """Convert a parameter value to string for XML embedding."""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def tool_calls_to_minimax_xml(tool_calls: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """Render OpenAI-format tool calls as MiniMax XML blocks."""
    if not tool_calls:
        return None

    lines: List[str] = ["<minimax:tool_call>"]

    for call in tool_calls:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        name = function.get("name") or call.get("name") or ""
        lines.append(f"<invoke name=\"{name}\">")

        arguments = function.get("arguments")
        parsed_args: Any = arguments

        if isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_args = arguments

        if isinstance(parsed_args, dict):
            for param_name, param_value in parsed_args.items():
                lines.append(f"<parameter name=\"{param_name}\">")
                lines.append(_format_param_value_for_xml(param_value))
                lines.append("</parameter>")
        else:
            lines.append("<parameter name=\"input\">")
            lines.append(_format_param_value_for_xml(parsed_args))
            lines.append("</parameter>")

        lines.append("</invoke>")

    lines.append("</minimax:tool_call>")
    return "\n".join(lines)


def extract_name(name_str: str) -> str:
    """Extract name from quoted string"""
    name_str = name_str.strip()
    if (name_str.startswith('"') and name_str.endswith('"')) or \
       (name_str.startswith("'") and name_str.endswith("'")):
        return name_str[1:-1]
    return name_str


def convert_param_value(value: str, param_type: str) -> Any:
    """Convert parameter value based on parameter type"""
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
        # Try JSON parsing, return string if failed
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value


def parse_tool_calls(model_output: str, tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Extract all tool calls from model output.

    Args:
        model_output: Complete output text from the model
        tools: Tool definition list for getting parameter type information

    Returns:
        {
            "tools_called": bool,
            "tool_calls": List[Dict],  # OpenAI format
            "content": str  # Content without tool call blocks
        }
    """
    # Quick check if tool call marker is present
    if "<minimax:tool_call>" not in model_output:
        return {
            "tools_called": False,
            "tool_calls": [],
            "content": model_output
        }

    tool_calls = []

    try:
        # Match all <minimax:tool_call> blocks (MiniMax may add whitespace around attributes)
        tool_call_regex = re.compile(r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL)
        invoke_regex = re.compile(r"<invoke\s+name\s*=\s*(.*?)</invoke>", re.DOTALL)
        parameter_regex = re.compile(r"<parameter\s+name\s*=\s*(.*?)</parameter>", re.DOTALL)

        # Iterate through all tool_call blocks
        for tool_call_match in tool_call_regex.findall(model_output):
            # Iterate through all invokes in this block
            for invoke_match in invoke_regex.findall(tool_call_match):
                # Extract function name
                name_match = re.search(r'^([^>]+)', invoke_match)
                if not name_match:
                    continue

                function_name = extract_name(name_match.group(1))

                # Get parameter configuration
                param_config = {}
                if tools:
                    for tool in tools:
                        tool_name = tool.get("name") or tool.get("function", {}).get("name")
                        if tool_name == function_name:
                            params = tool.get("parameters") or tool.get("function", {}).get("parameters")
                            if isinstance(params, dict) and "properties" in params:
                                param_config = params["properties"]
                            break

                # Extract parameters
                param_dict = {}
                for match in parameter_regex.findall(invoke_match):
                    param_match = re.search(r'^([^>]+)>(.*)', match, re.DOTALL)
                    if param_match:
                        param_name = extract_name(param_match.group(1))
                        param_value = param_match.group(2).strip()

                        # Remove leading and trailing newlines
                        if param_value.startswith('\n'):
                            param_value = param_value[1:]
                        if param_value.endswith('\n'):
                            param_value = param_value[:-1]

                        # Get parameter type and convert
                        param_type = "string"
                        if param_name in param_config:
                            if isinstance(param_config[param_name], dict) and "type" in param_config[param_name]:
                                param_type = param_config[param_name]["type"]

                        param_dict[param_name] = convert_param_value(param_value, param_type)

                # Build OpenAI-format tool call
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(param_dict, ensure_ascii=False)
                    }
                })

    except Exception as e:
        print(f"Failed to parse tool calls: {e}")
        return {
            "tools_called": False,
            "tool_calls": [],
            "content": model_output
        }

    if not tool_calls:
        return {
            "tools_called": False,
            "tool_calls": [],
            "content": model_output
        }

    # Extract content without tool call blocks
    content_regex = re.compile(r"<minimax:tool_call>.*?</minimax:tool_call>", re.DOTALL)
    content = content_regex.sub('', model_output).strip()

    return {
        "tools_called": True,
        "tool_calls": tool_calls,
        "content": content if content else None
    }
