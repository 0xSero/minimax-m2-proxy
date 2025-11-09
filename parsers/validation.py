"""Tool call validation and repair"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def validate_and_repair_tool_calls(
    tool_calls: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None
) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Validate tool calls and attempt to repair common issues.

    Returns:
        (repaired_tool_calls, warnings)
    """
    if not tool_calls:
        return [], []

    repaired = []
    warnings = []

    for tc in tool_calls:
        function_name = tc.get("function", {}).get("name")
        arguments_str = tc.get("function", {}).get("arguments", "{}")

        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError:
            warnings.append(f"Tool {function_name}: Invalid JSON arguments")
            continue

        # Find tool schema
        tool_schema = None
        if tools:
            for tool in tools:
                tool_name = tool.get("name") or tool.get("function", {}).get("name")
                if tool_name == function_name:
                    tool_schema = tool.get("parameters") or tool.get("function", {}).get("parameters")
                    break

        if not tool_schema:
            # No schema to validate against, pass through
            repaired.append(tc)
            continue

        # Check required parameters
        required_params = tool_schema.get("required", [])
        properties = tool_schema.get("properties", {})
        missing_params = []
        empty_params = []

        for param in required_params:
            if param not in arguments:
                missing_params.append(param)
            elif arguments[param] == "" or arguments[param] is None:
                empty_params.append(param)

        # Log issues
        if missing_params:
            param_list = ", ".join(missing_params)
            warning = f"Tool {function_name}: Missing required parameters: {param_list}"
            warnings.append(warning)
            logger.warning(warning)

        if empty_params:
            param_list = ", ".join(empty_params)
            warning = f"Tool {function_name}: Empty required parameters: {param_list}"
            warnings.append(warning)
            logger.warning(warning)

        # Decision: Skip tool calls with missing/empty required params
        if missing_params or empty_params:
            logger.error(f"Dropping malformed tool call for {function_name}")
            continue

        # Valid tool call
        repaired.append(tc)

    return repaired, warnings


def should_retry_with_stronger_prompt(warnings: List[str]) -> bool:
    """
    Determine if we should retry the request with a stronger tool calling prompt.

    This could be used in a retry loop, but for now just returns True if there are warnings.
    """
    return len(warnings) > 0
