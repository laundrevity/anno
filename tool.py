import inspect
import json
from typing import Any, get_type_hints, Union
import re

# Maps common Python types to JSON schema "type" keywords.
# Feel free to extend/adjust this mapping as needed.
python_type_to_json_type = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

def _parse_docstring(docstring: str) -> tuple[str, dict[str, str]]:
    """
    Split the docstring into two parts:
      1) The top-level (description) portion
      2) A dictionary of parameter descriptions keyed by parameter name
    """
    if not docstring:
        return ("", {})

    # Normalize line breaks, remove extra indentation
    lines = [line.strip() for line in docstring.strip().splitlines()]

    description_lines = []
    param_descriptions = {}

    # Simple state machine: reading "description" until we see ":param"
    in_params_section = False

    for line in lines:
        if line.startswith(":param"):
            # Mark that we are in param lines
            in_params_section = True

            # Try to capture  :param <name>: <description> 
            match = re.match(r":param\s+(\w+)\s*:\s*(.*)", line)
            if match:
                param_name, param_desc = match.groups()
                param_descriptions[param_name] = param_desc
        else:
            if not in_params_section:
                description_lines.append(line)
            else:
                # If we are in param lines, but the line does not start
                # with :param, it might be a continuation of a param description
                if param_descriptions:
                    # Last inserted param is the "active" one
                    last_param = list(param_descriptions.keys())[-1]
                    param_descriptions[last_param] += " " + line

    # Join the top-level description
    top_description = " ".join(description_lines)
    return top_description, param_descriptions


def _python_type_to_json_schema(py_type: Any) -> Union[str, list[str]]:
    """
    Convert a Python type annotation into a JSON schema type descriptor.
    For unions that include None, returns a list that includes "null".
    Example: int | None -> ["integer", "null"]
    """
    # Handle type aliases introduced by '|' (Python 3.10+),
    # e.g. int | None -> types.UnionType
    if str(py_type).startswith("typing.Union") or getattr(py_type, "__origin__", None) is Union:
        # For older python versions we might need to check __args__ for 'NoneType'
        args = getattr(py_type, "__args__", [])
        # Flatten all sub-arguments for type union
        types_list = []
        for arg in args:
            # Recursively convert each argument to a JSON type
            sub = _python_type_to_json_schema(arg)
            if isinstance(sub, list):
                types_list.extend(sub)
            else:
                types_list.append(sub)
        # Remove duplicates
        return list(set(types_list))

    # For the builtin "NoneType" (when it's not inside a union),
    # we just say "null"
    if py_type is type(None):  # noqa: E721
        return "null"

    # For generics like list[str], dict[str, int], etc., you might want to
    # handle them specifically. For simplicity, we’ll just call them "array"
    # or "object" and skip nested details. Adjust if you need more detail:
    origin = getattr(py_type, "__origin__", None)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"

    # Simple direct mapping
    return python_type_to_json_type.get(py_type, "string")  # default fallback


def tool(func):
    """
    Decorator that inspects the wrapped function and builds a
    Function Calling JSON schema as the `tool` attribute on the function.
    """
    # Extract name
    func_name = func.__name__

    # Extract docstring
    docstring = func.__doc__ or ""

    # Parse docstring to separate main description from param descriptions
    top_description, param_descriptions = _parse_docstring(docstring)

    # Extract signature to see which params have defaults
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        # If there's a type hint, use it; otherwise default to "string"
        py_type = hints.get(param_name, str)
        json_type = _python_type_to_json_schema(py_type)

        # Attempt to retrieve the docstring text for this param
        desc_for_param = param_descriptions.get(param_name, "")

        # Build the property object
        prop = {
            "type": json_type
        }
        if desc_for_param:
            prop["description"] = desc_for_param

        properties[param_name] = prop

        # If parameter has no default, mark it as required
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        # If the default is None, it usually means it's optional.
        # So we do NOT add it to "required" in that case.

    # Build the final JSON structure
    # "type": "function" is optional, but matches the structure in the example
    schema = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": top_description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
                "additionalProperties": False
            },
            "strict": True
        }
    }

    # Attach JSON schema to the function so we can reference it
    func._tool = schema

    return func
